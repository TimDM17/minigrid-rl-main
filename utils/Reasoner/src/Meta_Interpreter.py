import numpy as np

import re


VARIABLE_REGEX = r"^[A-Z_][A-Za-z0-9_]*$"
def find_solutions(rules_text, query_text,lang, clauses,):
    """Parse the query text and use our database rules to search for matching
    query solutions. """
    VARIABLE_REGEX = r"^[A-Z_][A-Za-z0-9_]*$"
    query_argument, query_pred, str_query_argument = Parser(query_text)

    if determinate_clause(clauses, query_pred):
        return evaluate_clauses(rules_text,lang, clauses, query_pred,query_argument )
    else:
        return evaluate_predicate(rules_text,query_argument, query_pred, str_query_argument)

def evaluate_predicate(rules_text,query_argument, query_pred, str_query_argument):

    variable_in_query = False

    if re.match(VARIABLE_REGEX, query_argument[1][1::]):
        query_argument[1] = query_argument[1][1::]
        variable_in_query = True
    if re.match(VARIABLE_REGEX, query_argument[0]):
        variable_in_query = True

    matching_query_terms = [item for item in rules_pred(query_pred, rules_text)]

    'first rule and query qith variables'
    if matching_query_terms:

        if variable_in_query:
            solutions_map = []

            for item in matching_query_terms:
                query_variable_map = {}

                for j in range(len(query_argument)):
                    if re.match(VARIABLE_REGEX, query_argument[j]):
                        del_query = np.delete( query_argument,j)
                        terms_namees = str(item.terms).split(',')
                        terms_namees[0] = terms_namees[0][1::]
                        terms_namees[-1] = terms_namees[-1][:-1]
                        terms = [str(at) for at in terms_namees]

                        if re.match(VARIABLE_REGEX, del_query[0]):
                            query_variable_map[query_argument[j]] = terms[j]
                        else:
                            del_term = np.delete(terms, j)
                            if del_query == del_term:
                                query_variable_map[query_argument[j]] = terms[j]

                solutions_map.append(query_variable_map)
            del_index = []
            for i in range(len(solutions_map)):
                if solutions_map[i] == {}:
                    del_index.append(i)
            solutions_map = np.delete(solutions_map,del_index)

            return  solutions_map
        else:
            if not variable_in_query:
                evaluated_value = []
                str_query_argument = '('
                for i in range(0, len(query_argument)-1):
                    str_query_argument +=   str(query_argument[i])+','
                str_query_argument += str(query_argument[-1]) +')'
                for item in matching_query_terms :

                    evaluated_value.append(1) if str_query_argument == str(item.terms) else evaluated_value.append(0)

            return bool(sum(evaluated_value))

'determinate if the query is a clause'
def determinate_clause(clauses, query_pred):
    head_lst = [i.head.pred.name for i in clauses]
    index = False
    for i in range(len(head_lst)):
        if head_lst[i] == query_pred:
            index = i

    if index == False:
        return False
    else:
        return True

'get clause from query'
def evaluate_clauses( rules_text, lang, clauses, query_pred,query_argument):
    head_lst = [i.head.pred.name for i in clauses]

    for i in range(len(head_lst)):
        if head_lst[i] == query_pred:
            index = i

    bodys = clauses[index].body
    terms_in_head = clauses[index].head.terms

    evaluate_lst = []
    body_for_query = []
    for j in range(len(bodys)):
        body_for_query.append(bodys[j].pred.name+':')
        for i in range(len(bodys[j].terms)):

            for m in range(len(terms_in_head)):
                if terms_in_head[m] == bodys[j].terms[i]:
                    index = m
            body_for_query[j] += query_argument[index] +','
        body_for_query[-1] = body_for_query[-1][:-1]
        body_argument, body_pred, str_body_argument = Parser_clause(body_for_query[j])
        evaluate_lst.append(evaluate_predicate(rules_text,body_argument, body_pred, str_body_argument))

    if all(isinstance(n, bool) for n in evaluate_lst):
        if sum(evaluate_lst) == len(evaluate_lst):
            return True
        else:
            return False

    elif any(isinstance(n, bool) for n in evaluate_lst):
        result =  []
        for item in evaluate_lst:
            if not isinstance(item, bool):
                result.append(item)
        return result

    else:
        shared_quan = [len((item)) for item in evaluate_lst]

        index = np.argmin(shared_quan)
        evaluate_other_lst = np.delete(evaluate_lst, index)
        solution = []
        for item in evaluate_lst[index]:
            values = list(item.values())[0]

            if 'obj' in values:
                if np.min(shared_quan)==4:
                    for item in evaluate_lst:
                        solution.append(item)
                else:


                    for item_other in evaluate_other_lst:
                        for i in range(len(item_other)):
                            aa = set(item.items())  # {('a', 1), ('b', 2)}
                            bb = set(item_other[i].items())
                            if aa.issubset(bb):
                                solution.append(bb)
            else:
                for item in evaluate_lst:
                    solution.append(item)


        shared_term_compare_with_other_evaluation_lst = []
#        temp = []
#        res = dict()
#        for items in solution:
#            for item in items:
#                for key, val in item:
#                    if val not in temp:
#                        temp.append(key)
#                        res[key] = val
        return  solution






'get the pred of the query from the all included pred  in rule_text'
def rules_pred(query_pred, rules_text):
    rules_pred = np.zeros(rules_text.shape).astype('object')
    for i in range(1, rules_pred.shape[0]):

        rules_pred[i] = rules_text[i].pred.name

    called_pred = np.where(rules_pred == query_pred )
    called_truth = rules_text[called_pred]
    return called_truth

'analyse text to get what exactly the question wants to get as answer'
def Parser(text):

    pred, argument_names_str = text.split('(')
    argument_names_str = argument_names_str[:-1]

    argument_names = argument_names_str.split(',')
    argument = [at for at in argument_names]


    return argument, pred, argument_names_str

def Parser_clause(text):

    pred, argument_names_str = text.split(':')

    argument_names = argument_names_str.split(',')
    argument = [at for at in argument_names]


    return argument, pred, argument_names_str