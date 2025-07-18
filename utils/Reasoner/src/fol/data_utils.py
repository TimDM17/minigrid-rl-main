import os.path

from lark import Lark
from .exp_parser import ExpTree
from .language import Language, DataType
from .logic import Predicate, NeuralPredicate, PlannerPredicate, FuncSymbol, Const


class DataUtils(object):
    """Utilities about logic.

    A class of utilities about first-order logic.

    Args:
        dataset_type (str): A dataset type (kandinsky or clevr).
        dataset (str): A dataset to be used.

    Attrs:
        base_path: The base path of the dataset.
    """

    def __init__(self, lark_path, lang_base_path, dataset_type='kandinsky', dataset='two_doors'):
        self.base_path = lang_base_path + dataset_type + '/' + dataset + '/'
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_atom = Lark(grammar.read(), start="atom")
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_clause = Lark(grammar.read(), start="clause")

    def load_clauses(self, path, lang):
        """Read lines and parse to Atom objects.
        """
        clauses = []
        with open(path) as f:
            for line in f:
                tree = self.lp_clause.parse(line[:-1])
                clause = ExpTree(lang).transform(tree)
                clauses.append(clause)
        return clauses

    def load_atoms(self, path, lang):
        """Read lines and parse to Atom objects.
        """
        atoms = []

        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    tree = self.lp_atom.parse(line[:-2])
                    atom = ExpTree(lang).transform(tree)
                    atoms.append(atom)
        return atoms

    def load_preds(self, path):
        f = open(path)
        lines = f.readlines()
        preds = [self.parse_pred(line) for line in lines]
        return preds

    def load_neural_preds(self, path):
        f = open(path)
        lines = f.readlines()
        preds = [self.parse_neural_pred(line) for line in lines]
        return preds

    def load_move_preds(self, path):
        f = open(path)
        lines = f.readlines()
        preds = [self.parse_move_pred(line) for line in lines]
        return preds

    def load_consts(self, path):
        f = open(path)
        lines = f.readlines()
        consts = []
        for line in lines:
            consts.extend(self.parse_const(line))
        return consts

    def load_consts_mi(self, path):
        f = open(path)
        lines = f.readlines()
        consts = []
        for line in lines:
            consts.extend(self.parse_const_mi(line))
        return consts
    def load_funcs(self, path):
        funcs = []
        if os.path.isfile(path):
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    funcs.append(self.parse_func(line))
        return funcs

    def parse_pred(self, line):
        """Parse string to predicates.
        """
        line = line.replace('\n', '')
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        assert int(arity) == len(
            dtypes), 'Invalid arity and dtypes in ' + pred + '.'
        return Predicate(pred, int(arity), dtypes)

    def parse_neural_pred(self, line):
        """Parse string to predicates.
        """
        line = line.replace('\n', '')
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        assert int(arity) == len(
            dtypes), 'Invalid arity and dtypes in ' + pred + '.'
        return NeuralPredicate(pred, int(arity), dtypes)

    def parse_move_pred(self, line):
        """Parse string to predicates.
        """
        line = line.replace('\n', '')
        pred, arity, dtype_names_str = line.split(':')
        dtype_names = dtype_names_str.split(',')
        dtypes = [DataType(dt) for dt in dtype_names]
        assert int(arity) == len(
            dtypes), 'Invalid arity and dtypes in ' + pred + '.'
        return PlannerPredicate(pred, int(arity), dtypes)

    def parse_func(self, line):
        """Parse string to function symbols.
        """
        name, arity, in_dtypes, out_dtype = line.replace("\n", "").split(':')
        in_dtypes = in_dtypes.split(',')
        in_dtypes = [DataType(in_dtype) for in_dtype in in_dtypes]
        out_dtype = DataType(out_dtype)
        return FuncSymbol(name, int(arity), in_dtypes, out_dtype)

    def parse_const(self, line):
        """Parse string to function symbols.
        """
        line = line.replace('\n', '')
        dtype_name, const_names_str = line.split(':')
        dtype = DataType(dtype_name)
        const_names = const_names_str.split(',')
        return [Const(const_name, dtype) for const_name in const_names]

    def parse_const_mi(self, line):
        """Parse string to function symbols.
        """
        line = line.replace('\n', '')
        line = line[:-1]
        dtype_name, const_names_str = line.split(':')
        dtype = DataType(dtype_name)
        const_names = const_names_str.split(';')
        return [Const(const_name, dtype) for const_name in const_names]

    def parse_clause(self, clause_str, lang):
        tree = self.lp_clause.parse(clause_str)
        return ExpTree(lang).transform(tree)

    def get_clauses(self, lang):
        return self.load_clauses(self.base_path + 'clauses.txt', lang)

    def get_clauses_mi(self, lang):
        return self.load_clauses(self.base_path + 'mi_clauses.txt', lang)

    def get_bk(self, lang):
        return self.load_atoms(self.base_path + 'bk.txt', lang)

    def load_language(self):
        """Load language, background knowledge, and clauses from files.
        """
        meta_preds = self.load_preds(self.base_path + 'meta_preds.txt')
        preds = self.load_preds(self.base_path + 'preds.txt') + \
            self.load_neural_preds(self.base_path + 'neural_preds.txt')
        consts = self.load_consts(self.base_path + 'consts.txt')
        funcs = self.load_funcs(self.base_path + 'funcs.txt')
        lang = Language(preds, funcs, consts)

        return lang

    def tranform_object_clause_into_preconditions_postconditions_actions(self, object_clause):
        # Z: batched output
        precondition = []
        postcondition = []
        action = []

        for clause in object_clause:
            precondition.append(str(clause.body[0]))
            postcondition.append(str(clause.head))
            action.append(str(clause.body[1]))
        currentstate = set(precondition + postcondition)
        goal = str(object_clause[-1].head)
        return precondition, postcondition, action, currentstate, goal

    def generate_const_for_mi(self,atoms, lang, clauses):
        atoms = atoms[2::]
        precondition, postcondition, action, currentstate, goal = self.tranform_object_clause_into_preconditions_postconditions_actions(clauses)
        with open(self.base_path + 'mi_consts.txt', 'w') as f:
            f.write('position:')
            for item in atoms:
                if type(item.pred ) in [NeuralPredicate,PlannerPredicate]:
                    f.write(str(item) + ";")
            f.write('\n'+'precondition:')
            for item in precondition:
                f.write(item + ";")
            f.write('\n'+'postcondition:')
            for item in postcondition:
                f.write(item + ";")
            f.write('\n' + 'currentstate:')
            for item in currentstate:
                f.write(item + ";")
            f.write('\n'+'movenamer:')
            for item in action:
                f.write(item + ";")
            f.write('\n' + 'goal:')
            f.write(goal+ ";")
            f.write('\n' + 'startstate:')
            f.write('initial(A);')
            f.write('\n' + 'movenamesr:')
            f.write('*;')
    def load_language_mi(self,obj_atoms):
        """Load language, background knowledge, and clauses from files.
        """
        a=0
        b =a
        preds = self.load_preds(self.base_path + 'meta_preds.txt')+ \
            self.load_neural_preds(self.base_path + 'mi_preds.txt')
        consts = self.load_consts_mi(self.base_path + 'mi_consts.txt')
        funcs = self.load_funcs(self.base_path + 'funcs_mi.txt')
        lang = Language(preds, funcs, consts)

        return lang
