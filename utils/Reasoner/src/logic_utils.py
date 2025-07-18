from .infer import InferModule, InferModule_mi
from .tensor_encoder import TensorEncoder, TensorEncoder_mi
from .fol.logic import *
from .fol.data_utils import DataUtils
from .fol.language import DataType


p_ = Predicate('.', 1, [DataType('spec')])
false = Atom(p_, [Const('__F__', dtype=DataType('spec'))])
true = Atom(p_, [Const('__T__', dtype=DataType('spec'))])


def get_lang(lark_path, lang_base_path, dataset_type, dataset):
    """Load the language of first-order logic from files.

    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language.
    """
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path,
                   dataset_type=dataset_type, dataset=dataset)
    lang = du.load_language()
    clauses = du.get_clauses(lang)
#    bk = du.get_bk(lang)
    atoms = generate_atoms(lang)
    du.generate_const_for_mi(atoms, lang, clauses)
    return lang, clauses, atoms
def get_mi_lang(obj_clauses, obj_atoms, lark_path, lang_base_path, dataset_type, dataset):
    du = DataUtils(lark_path=lark_path, lang_base_path=lang_base_path,
                  dataset_type=dataset_type, dataset=dataset)
    lang_mi = du.load_language_mi(obj_atoms)
    clauses_mi = du.get_clauses_mi(lang_mi)
    a = clauses_mi

    atoms_mi = generate_atoms_mi(lang_mi, obj_clauses)

    return lang_mi, clauses_mi,  atoms_mi


def  build_infer_module_mi(clauses, atoms, lang, device, m=3, infer_step=3, train=False):
    te = TensorEncoder_mi(lang, atoms, clauses, device=device)
    I = te.encode()
    im = InferModule_mi(I, m=m, infer_step=infer_step, device=device, train=train)
    return im





def  build_infer_module(clauses, atoms, lang, device, m=3, infer_step=3, train=False):
    te = TensorEncoder(lang, atoms, clauses, device=device)
    I = te.encode()
    im = InferModule(I, m=m, infer_step=infer_step, device=device, train=train)
    return im

def generate_terms(lang, max_term_depth):
    consts = lang.consts
    terms = consts
    funcs = lang.funcs
    #terms = []
    for f in funcs:
        if str(f)[0] in ['f','l', 'u', 'd']:
            max = 1
        elif str(f)[0] in ['r']:
            max = 3
        else:
            max = max_term_depth
        new_terms = []
        for i in range(max):
        #print(terms)
            terms_list = []
            for in_dtype in f.in_dtypes:
                terms_dt = [term for term in terms if term.dtype == in_dtype]
                if terms_dt:
                    terms_list.append(terms_dt)
            #if len(terms_list)==1:
            #    terms_list.append(terms_list[0])
            args_list = list(set(itertools.product(*terms_list)))
            for args in args_list:
                new_terms.append(FuncTerm(f, args))
            new_terms = list(set(new_terms))
            terms.extend(new_terms)
    return list(terms)

def tranform_object_clause_into_move_atoms(object_clause):
    # Z: batched output
    a = []
    for clause in object_clause:
        b = str('move') + '(' + str(clause.body[1]) + ',' + str(clause.body[0]) + ',' + str(
            clause.head) + ')'
        a.append(b)
    return a

def has_duplicates(lst):
    """
    Function to check if a list contains duplicate elements.
    :param lst: List of elements
    :return: True if duplicates exist, False otherwise
    """
    seen = set()
    for item in lst.all_consts():
        if item in seen:
            return True
        seen.add(item)
    return False

def is_functor(element):
    """
    Function to check if an element is a functor (for example, a specific type like list, tuple, or a custom class).
    :param element: Element to check
    :return: True if element is a functor, False otherwise
    """
    return str(element.dtype)=='movenamesr'  # Modify this based on your definition of a functor


def is_valid_sequence(seq):
    valid_patterns = [
        ["gtrd(A,C)", "grk(A,B)", "*"],
        ["grk(A,B)", "*"],
        ["*"]
        ["gtbd(A,C)", "*"]
    ]

    for pattern in valid_patterns:
        if len(seq) >= len(pattern) and seq[-len(pattern):] == pattern:
            return True

    return False


def filter_lists(lists):
    filtered_list = []
    for i, element in enumerate(lists):
        if is_functor(element):
            if is_valid_sequence(element):
                filtered_list.append(element)
        else:
            filtered_list.append(element)
    return [seq for seq in lists if is_valid_sequence(seq)]
def filter_functors(main_list):
    """
    Function to filter out functor elements that contain duplicates from the last several elements of the main list.
    :param main_list: List containing mixed data types with functors at the end
    :return: Main list with only functors without duplicates kept
    """
    filtered_list = []
    for i, element in enumerate(main_list):
        if is_functor(element):
            if not has_duplicates(element):
                filtered_list.append(element)
        else:
            filtered_list.append(element)

    return filtered_list

def generate_atoms_mi(lang, obj_clauses, max_term_depth=1):
    spec_atoms = [false, true]
    atoms = []
    bk = tranform_object_clause_into_move_atoms(obj_clauses)
    terms = generate_terms(lang, max_term_depth=max_term_depth)
    terms = filter_functors(terms)
   #further reduce the number of the functors
  #  terms = filter_lists(terms)

    for pred in lang.preds:

        dtypes = pred.dtypes
        terms_list = [[term for term in terms if term.dtype == dtype] for dtype in dtypes]
        args_list = list(set(itertools.product(*terms_list)))

        #for this predicate we need to generate all possible atoms
        if pred.name == 'condition_met' or pred.name == 'equal':# or pred.name == 'plan':
            for args in args_list:
                atoms.append(Atom(pred, args))
        elif pred.name=='plan':
            for args in args_list:
                #if str(args[1])=='initial(A)' and str(args[3])=='*':
                atoms.append(Atom(pred, args))
                #elif str(args[1])=='get(A,B)' and str(args[3])=='r(gk(A,B),*)':
                #    atoms.append(Atom(pred, args))
        elif str(pred) == 'move/3/[movenamer, precondition, postcondition]':
            for args in args_list:
                if len(args) == 1 or len(set(args)) == len(args):
                    if str(Atom(pred, args)) in bk:
                        atoms.append(Atom(pred, args))
        else:
            for args in args_list:
                if len(args) == 1 or len(set(args)) == len(args):
                    # if len(args) == 1 or (args[0] != args[1] and args[0].mode == args[1].mode):
                    # if len(set(args)) == len(args):
                    # if not (str(sorted([str(arg) for arg in args])) in args_str_list):
                    atoms.append(Atom(pred, args))
                    # args_str_list.append(
                    #    str(sorted([str(arg) for arg in args])))
                    # print('add atom: ', Atom(pred, args))
    return spec_atoms + sorted(atoms)
def generate_atoms(lang, max_term_depth = 5):
    spec_atoms = [false, true]
    atoms = []
    terms = generate_terms(lang, max_term_depth=max_term_depth)

    for pred in lang.preds:

        dtypes = pred.dtypes
        terms_list = [[term for term in terms if term.dtype == dtype] for dtype in dtypes]
        # consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
        args_list = list(set(itertools.product(*terms_list)))
        # args_list = lang.get_args_by_pred(pred)
        args_str_list = []
        # args_mem = []
        for args in args_list:
            if len(args) == 1 or len(set(args)) == len(args):
                # if len(args) == 1 or (args[0] != args[1] and args[0].mode == args[1].mode):
                # if len(set(args)) == len(args):
                # if not (str(sorted([str(arg) for arg in args])) in args_str_list):
                atoms.append(Atom(pred, args))
                # args_str_list.append(
                #    str(sorted([str(arg) for arg in args])))
                # print('add atom: ', Atom(pred, args))
    'generate atom for planning'
 #   planner_atoms = [move(obj0, up), move(obj0, down), move(obj0, left), move(obj0, right) ]
    return spec_atoms + sorted(atoms)#+planner_atoms


def generate_bk(lang):
    atoms = []
    for pred in lang.preds:
        if pred.name in ['diff_color', 'diff_shape']:
            dtypes = pred.dtypes
            consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
            args_list = itertools.product(*consts_list)
            for args in args_list:
                if len(args) == 1 or (args[0] != args[1] and args[0].mode == args[1].mode):
                    atoms.append(Atom(pred, args))
    return atoms


def get_index_by_predname(pred_str, atoms):
    for i, atom in enumerate(atoms):
        if atom.pred.name == pred_str:
            return i
    assert 1, pred_str + ' not found.'


def parse_clauses(lang, clause_strs):
    du = DataUtils(lang)
    return [du.parse_clause(c) for c in clause_strs]
