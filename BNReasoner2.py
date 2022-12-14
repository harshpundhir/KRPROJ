import pandas as pd
from typing import Union, Literal, List
from BayesNet import BayesNet
import itertools
import networkx as nx
from copy import deepcopy
import cProfile

import warnings

warnings.filterwarnings('ignore')


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    @staticmethod
    def leaves(bn: BayesNet) -> set[str]:
        vars = bn.get_all_variables()
        ls = set()
        for v in vars:
            if len(bn.get_children(v)) == 0:
                ls.add(v)
        return ls

    @staticmethod
    def has_undirected_path(bn: BayesNet, x: str, Y: set[str]):
        visited = [x]
        seen = set()
        while len(visited) > 0:
            node = visited.pop(0)
            seen.add(node)
            ancs = BNReasoner.parents(bn, node) - seen
            children = set(bn.get_children(node)) - seen
            if len(Y.intersection(children.union(ancs))) > 0:
                return True
            visited.extend(ancs)
            visited.extend(children)
        return False

    @staticmethod
    def parents(bn: BayesNet, x: str) -> set[str]:
        rents = set()
        for var in bn.get_all_variables():
            if x in bn.get_children(var):
                rents.add(var)
        return rents

    @staticmethod
    def disconnected(bn: BayesNet, X: set[str], Y: set[str]) -> bool:
        for x in X:
            if BNReasoner.has_undirected_path(bn, x, Y):
                return False
        for y in Y:
            if BNReasoner.has_undirected_path(bn, y, X):
                return False
        return True

    @staticmethod
    def marginalize_pr(cpt: pd.DataFrame, var: str, pr: float) -> pd.DataFrame:
        cpt.loc[cpt[var] == True, 'p'] = cpt[cpt[var] == True].p.multiply(pr)
        cpt.loc[cpt[var] == False, 'p'] = cpt[cpt[var] == False].p.multiply(1 - pr)
        all_other_vars = [col for col in cpt.columns if col not in [var, 'p']]
        cpt = cpt.groupby(all_other_vars, as_index=False).sum()
        del cpt[var]
        return cpt

    def num_deps(self, x):
        return len(self.bn.get_cpt(x).columns) - 2

    def _pr(self, x: str):
        """
        Computes probability of x Pr(x) in graph by marginalizing on
        dependencies
        """
        cpt = self.bn.get_cpt(x)
        deps = [col for col in cpt.columns if col not in ['p', x]]
        if len(deps) == 0:
            return cpt[cpt[x] == True].p.values[0]
        cpt = cpt.copy()
        for dep in deps:
            pr_dep = self._pr(dep)
            cpt = BNReasoner.marginalize_pr(cpt, dep, pr_dep)
        return cpt[cpt[x] == True].p.values[0]

    def prune(self, Q: set[str], e: set):
        """
        Given a set of query variables Q and evidence e, node- and edge-prune the
        Bayesian network s.t. queries of the form P(Q|E) can still be correctly
        calculated. (3.5 pts)
        """
        # e_vars = set(e.index)
        # Delete all edges outgoing from evidence e and replace with reduced factor
        for e_var in e:
            children = self.bn.get_children(e_var)
            for child in children:
                self.bn.del_edge((e_var, child))  # Edge prune
                child_cpt = self.bn.get_cpt(child)
                reduced = BayesNet.reduce_factor(e, child_cpt)  # Replace child cpt by reduced cpt
                reduced = reduced[reduced.p > 0.]  # Delete any rows where probability is 0
                self.bn.update_cpt(child, reduced)

        # Node pruning after edge pruning
        leaves = BNReasoner.leaves(self.bn) - set(e_vars.union(Q))
        while len(leaves) > 0:
            leaf = leaves.pop()
            self.bn.del_var(leaf)
            leaves = leaves.union(BNReasoner.leaves(self.bn) - set(e_vars.union(Q)))

    def dsep(self, X: set[str], Y: set[str], Z: set[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is d-separated
        of Y given Z. (4pts)
        """
        bn = deepcopy(self.bn)
        while True:
            leaves = BNReasoner.leaves(bn) - X.union(Y).union(Z)
            had_leaves = bool(len(leaves))
            zout_edges = []
            for z in Z:
                zout_edges.extend((z, child) for child in bn.get_children(z))
            had_edges = bool(len(zout_edges))
            if not had_edges and not had_leaves:
                break
            for u, v in zout_edges:
                bn.del_edge((u, v))
            for leaf in leaves:
                bn.del_var(leaf)
        return BNReasoner.disconnected(bn, X, Y)

    def independent(self, X, Y, Z):
        """
        Independence: Given three sets of variables X, Y, and Z, determine whether X
        is independent of Y given Z. (Hint: Remember the connection between d-separation
        and independence) (1.5pts)
        """
        return self.dsep(X, Y, Z)

    def _compute_max_trivial(self, original_ft: pd.DataFrame, ft, x):
        row = original_ft.iloc[original_ft.p.idxmax()].copy()
        x_assignment = row[x]
        del row[x]
        row[f'instantiation_{x}'] = x_assignment
        ft.loc[0, :] = row
        return ft

    def _compute_sum_trivial(self, original_ft, ft, x):
        ft.loc[0, 'p'] = original_ft.p.sum()
        return ft

    def _compute_new_cpt(self, factor_table, x, which):
        """
        Given a factor and a variable X, compute the CPT in which X is either summed-out or maxed-out. (3pts)
        """
        columns = [col for col in factor_table.columns if col != 'p' and 'instantiation_' not in col and x != col]
        inst_columns = [col for col in factor_table.columns if col.startswith('instantiation_')]
        f = pd.DataFrame(columns=[*columns, 'p', *inst_columns])
        if which == "max":
            f[f"instantiation_{x}"] = []

        if len(columns) == 0:
            if which == 'max':
                return self._compute_max_trivial(factor_table, f, x)
            else:
                return self._compute_sum_trivial(factor_table, f, x)
        instantiations = factor_table.groupby(columns).sum().index.values
        if len(instantiations) == 0:
            return factor_table
        if type(instantiations[0]) != type(tuple()):
            instantiations = list(map(lambda x: [x], instantiations))
        count = 0

        for inst in instantiations:
            inst_dict = {}
            inst_list = []
            for j in range(len(columns)):
                inst_dict[columns[j]] = inst[j]
                inst_list.append(inst[j])
            inst_series = pd.Series(inst_dict)
            comp_inst = self.bn.get_compatible_instantiations_table(inst_series, factor_table)
            if which == 'max':
                new_p = comp_inst.p.max()
                row = comp_inst.loc[comp_inst["p"] == new_p]
                inst_list.append(new_p)
                for inst_column in inst_columns:
                    inst_list.append(row[inst_column].values[0])
                inst_list.append(row[x].values[0])
            elif which == 'sum':
                new_p = comp_inst.p.sum()
                inst_list.append(new_p)

            f.loc[count] = inst_list

            count += 1
        return f

    def marginalize(self, factor, x):
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out. (3pts)
        """
        return self._compute_new_cpt(self.bn.get_cpt(factor), x, 'sum')

    def maxing_out(self, factor: str, x):
        """
        Given a factor and a variable X, compute the CPT in which X is maxed-out. Remember
        to also keep track of which instantiation of X led to the maximized value. (5pts)

        """
        return self._compute_new_cpt(self.bn.get_cpt(factor), x, 'max')

    def _mult_as_factor(self, factor_table_f, factor_table_g):
        X = pd.DataFrame(columns=factor_table_f.columns)
        Y = pd.DataFrame(columns=factor_table_g.columns)

        # cols = [*[c for c in factor_table.columns if c != 'p'], 'p']
        columns_x = [col for col in X.columns if col != 'p' and 'instantiation_' not in col]
        columns_y = [col for col in Y.columns if col != 'p' and 'instantiation_' not in col and col not in columns_x]
        inst_columns_x = [col for col in X.columns if col.startswith('instantiation_')]
        inst_columns_y = [col for col in Y.columns if col.startswith('instantiation_')]
        df = pd.DataFrame(columns=[*columns_x, *columns_y, 'p', *inst_columns_x, *inst_columns_y])

        vars = [*columns_x, *columns_y]
        inst_cols = [*inst_columns_x, *inst_columns_y]
        l = [False, True]
        instantiations = [list(i) for i in itertools.product(l, repeat=len(vars))]

        for count, i in enumerate(instantiations):
            # for j in range(inst_df):
            f = {}
            g = {}
            for n, variable in enumerate(vars):
                if variable in factor_table_f.columns:
                    f[variable] = i[n]
                if variable in factor_table_g.columns:
                    g[variable] = i[n]
            f_series = pd.Series(f)
            g_series = pd.Series(g)

            comp_inst_f = self.bn.get_compatible_instantiations_table(f_series, factor_table_f)
            comp_inst_g = self.bn.get_compatible_instantiations_table(g_series, factor_table_g)

            value = comp_inst_f.p.sum() * comp_inst_g.p.sum()

            row = []
            for n, var in enumerate(vars):
                row.append(i[n])

            row.append(value)

            for inst in inst_cols:
                if inst in comp_inst_f.columns:
                    row.append(comp_inst_f.loc[:, inst].values[0])
                else:
                    row.append(comp_inst_g.loc[:, inst].values[0])

            df.loc[count] = row

        return df

    def factor_mult(self, factor_f, factor_g):
        """
        Given two factors f and g, compute the multiplied factor h=fg. (5pts)
        """
        factor_table_f = self.bn.get_cpt(factor_f)
        factor_table_g = self.bn.get_cpt(factor_g)
        return self._mult_as_factor(factor_table_f, factor_table_g)

    def _get_elim_order(self, vars, elim_method):
        if elim_method == 'min_degree':
            return self.min_degree_ordering(vars)
        elif elim_method == 'min_fill':
            return self.minfill_ordering(vars)
        else:
            raise ValueError(f'elim_method {elim_method} not supported')

    def get_factors_using(self, var: str):
        cpts = self.bn.get_all_cpts()
        factors = set()
        for f, cpt in cpts.items():
            if var in cpt.columns:
                factors.add(f)
        return factors

    def _reduce_variable(self, var: str):
        """Once again trying to implement elim var"""
        neighbors = self.get_factors_using(var)
        if len(neighbors) == 0:
            return None, None
        cpt = None
        for neighbor in neighbors:  # Multiply all neighbors together in factor multiply
            if cpt is None:
                cpt = self.bn.get_cpt(neighbor)
                continue
            neighbor_cpt = self.bn.get_cpt(neighbor)
            cpt = self._mult_as_factor(cpt, neighbor_cpt)
        cpt = self._compute_new_cpt(cpt, var, 'sum')  # Sum out variable
        # cpt.p /= cpt.p.sum() # normalize for joint probability
        return cpt, neighbors

    def variable_elimination2(self, to_elem: List[str], heuristic: [str] = 'min_degree'):
        tau = pd.DataFrame()
        visited = set()
        if heuristic == 'min_fill':
            to_elem = self.minfill_ordering(to_elem)
        else:
            to_elem = self.min_degree_ordering(to_elem)

        for element in to_elem:
            if tau.empty:
                tau = self.bn.get_cpt(element)
                # print(element, tau)
            to_visit = set(self.bn.get_children(element)) - set(visited)
            for child in to_visit:
                tau = self._mult_as_factor(tau, self.bn.get_cpt(child))
                visited.add(child)
            tau = self._compute_new_cpt(tau, element, 'sum')
            visited.add(element)
        return tau

    def variable_elimination(self, X: set[str],
                             elim_method: Union[Literal['min_fill'], Literal['min_degree']] = 'min_degree'):
        """
        Variable Elimination: Sum out a set of variables by using variable elimination.
        (5pts)
        This does not return anything but leaves the BN in a pruned/factored state.
        """

        # ------

        to_eliminate = set(self.bn.get_all_variables()) - X
        to_eliminate = self._get_elim_order(to_eliminate, elim_method)
        while len(to_eliminate) > 0:
            var = to_eliminate.pop(0)
            cpt, neighbors = self._reduce_variable(var)
            fname = self._get_factor_name(cpt)
            for neighbor in neighbors:
                self.bn.del_var(neighbor)
            self.bn.add_var(fname, cpt)

    @staticmethod
    def _get_factor_name(cpt: pd.DataFrame):
        cols = sorted([c for c in cpt.columns if c != 'p'])
        return 'f_' + ','.join(cols)

    def min_fill_degree_ordering(self, X):
        """Given a set of variables X in the Bayesian network,
        compute a good ordering for the elimination of X based on the min-degree heuristics (2pts)
        and the min-fill heuristics (3.5pts). (Hint: you get the interaction graph ???for free???
        from the BayesNet class.)"""
        interaction_graph = self.bn.get_interaction_graph()
        order = []

        for i in range(len(X)):
            Lowest_degree = self.lowest_degree(interaction_graph, X)
            order.append(Lowest_degree)
            if Lowest_degree[i - 1]==Lowest_degree:
                self.min_degree_ordering(X)
            interaction_graph = self.remove_node_interaction(Lowest_degree, interaction_graph)
            X.remove(Lowest_degree)
        return order

    def min_degree_ordering(self, X):
        """Given a set of variables X in the Bayesian network,
        compute a good ordering for the elimination of X based on the min-degree heuristics (2pts)
        and the min-fill heuristics (3.5pts). (Hint: you get the interaction graph ???for free???
        from the BayesNet class.)"""
        interaction_graph = self.bn.get_interaction_graph()
        order = []

        for i in range(len(X)):
            Lowest_degree = self.lowest_degree(interaction_graph, X)
            order.append(Lowest_degree)
            interaction_graph = self.remove_node_interaction(Lowest_degree, interaction_graph)
            X.remove(Lowest_degree)
        return order

    def remove_node_interaction(self, node, graph):
        neighbours = [n for n in nx.neighbors(graph, node)]
        for neighbour in neighbours:
            copy_neighbours = deepcopy(neighbours)
            copy_neighbours.remove(neighbour)
            for i in copy_neighbours:
                if not (self.nodes_connected(neighbour, i, graph)):
                    graph.add_edge(neighbour, i)
        graph.remove_node(node)
        return graph

    def min_degree_fill_ordering(self, X):
        """Given a set of variables X in the Bayesian network,
        compute a good ordering for the elimination of X based on the min-degree heuristics (2pts)
        and the min-fill heuristics (3.5pts). (Hint: you get the interaction graph ???for free???
        from the BayesNet class.)"""
        interaction_graph = self.bn.get_interaction_graph()
        order = []

        for i in range(len(X)):
            Lowest_degree = self.lowest_degree(interaction_graph, X)
            order.append(Lowest_degree)
            if Lowest_degree[i - 1]==Lowest_degree:
                self.minfill_ordering(X)
            interaction_graph = self.remove_node_interaction(Lowest_degree, interaction_graph)
            X.remove(Lowest_degree)
        return order
    def minfill_ordering(self, X):
        interaction_graph = self.bn.get_interaction_graph()
        order = []

        for i in range(len(X)):
            lowest_fill = self.lowest_fill(interaction_graph, X)
            order.append(lowest_fill)
            interaction_graph = self.remove_node_interaction(lowest_fill, interaction_graph)
            X.remove(lowest_fill)
        return order

    def lowest_fill(self, graph, X):
        lowest_fill = 100
        name = "test"
        for var in X:
            amount = self.amount_added_interaction(var, graph)
            if amount < lowest_fill:
                lowest_fill = amount
                name = var
        return name

    def amount_added_interaction(self, x, graph):
        neighbours = [n for n in nx.neighbors(graph, x)]
        added_count = 0
        for neighbour in neighbours:
            copy_neighbours = deepcopy(neighbours)
            copy_neighbours.remove(neighbour)
            for i in copy_neighbours:
                if not (self.nodes_connected(neighbour, i, graph)):
                    added_count += 1
        return added_count / 2

    def nodes_connected(self, u, v, graph):
        return u in graph.neighbors(v)

    def lowest_degree(self, graph, X):
        lowest_degree = 100
        name = "test"

        for i in X:
            value = graph.degree[i]
            if value < lowest_degree:
                lowest_degree = value
                name = i
        return name

    def _reduce_all_factors(self, e: pd.Series):
        for var in self.bn.get_all_variables():
            cpt = self.bn.get_cpt(var)
            cpt = BayesNet.reduce_factor(e, cpt)
            self.bn.update_cpt(var, cpt)

    def marginal_distribution(self, Q, e, elim_method='min_degree'):
        """Given query variables Q and possibly empty evidence e,
        compute the marginal distribution P(Q|e). Note that Q is a subset of
        the variables in the Bayesian network X with Q ??? X but can also be Q = X. (2.5pts)"""
        vars_to_keep = Q.union(set(e.index))
        # self.prune(Q, e)
        self._reduce_all_factors(e)
        self.variable_elimination(vars_to_keep, elim_method)
        jptd = self.multiply_all_tables()  # joint probability distribution Pr(Q ^ e)
        ept = jptd
        for q in Q:  # Sum out all variabes in Q from Pr(Q ^ e) to obtain Pr(e)
            ept = self._compute_new_cpt(ept, q, 'sum')

        pre = BayesNet.get_compatible_instantiations_table(e, ept)
        if len(pre) != 1:
            return pre

        pre = pre.p.values[0]
        jptd = BayesNet.get_compatible_instantiations_table(e, jptd)
        jptd.p /= pre  # Compute Pr(Q ^ e) / Pr(e)
        return jptd

    def _assignments_from_ept(self, ept):
        row = ept.iloc[0]
        assignments = {}
        for i, val in row.items():
            if i.startswith('instantiation_'):
                var_key = i.replace('instantiation_', '')
                assignments[var_key] = val
        return pd.Series(assignments)

    def map(self, Q: set[str], e: pd.Series, ret_jptd=False, elim_method='min_degree'):
        """Compute the maximum a-posteriory instantiation + value of query variables Q, given a possibly empty evidence e. (3pts)"""
        # self.prune(Q, e)
        vars_to_keep = Q.union(set(e.index))
        self.variable_elimination(vars_to_keep, elim_method)
        jptd = self.multiply_all_tables()  # joint probability distribution Pr(Q ^ e)
        ept = jptd
        for q in Q:
            ept = self._compute_new_cpt(ept, q, 'max')
        ept = BayesNet.get_compatible_instantiations_table(e, ept)
        if len(ept) != 1:
            return ept
        return self._assignments_from_ept(ept)

    def get_remaining_elements_except_evidence(self, e: [set]) -> [set]:
        allvars = self.bn.get_all_variables()
        ans = set()
        for i in allvars:
            if i not in e:
                ans.add(i)
        return ans

    def mpe(self, e):
        Q = self.get_remaining_elements_except_evidence(e)
        # Start with edge pruning and node pruning
        # Get elimination order
        # maximize out for order
        # self.prune(Q,e)
        vars_to_keep = Q.union(set(e.index))
        self._reduce_all_factors(e)
        self.variable_elimination(vars_to_keep)  # elminate all variables we don't need
        for var in Q:
            # order = self.min_degree_ordering(set(self.bn.get_all_variables()) - vars_to_keep)
            # for var in order:
            list_childeren = self.bn.get_children(var)
            # breakpoint()
            for child in list_childeren:
                multiply_table = self.factor_mult(var, child)
                self.bn.update_cpt(child, multiply_table)
                table = self.maxing_out(child, var)
                self.bn.update_cpt(child, table)

            if len(list_childeren) == 0:
                table = self.maxing_out(var, var)
                self.bn.update_cpt(var, table)
            else:
                self.bn.del_var(var)
        end_table = self.multiply_all_tables()
        end_table = BayesNet.get_compatible_instantiations_table(e, end_table)
        return self._assignments_from_ept(end_table)

    def multiply_all_tables(self):
        all_tables = self.bn.get_all_cpts()
        all_tables_list = list(all_tables.values())
        end_table = all_tables_list[0]
        for i in range(1, len(all_tables_list)):
            end_table = self._mult_as_factor(end_table, all_tables_list[i])

        return end_table


def test_prune():
    reasoner = BNReasoner('testing/lecture_example.BIFXML')
    e = pd.Series({'Rain?': False})
    Q = {'Slippery Road?', 'Winter?'}
    print(reasoner.bn.get_all_variables())
    reasoner.prune(Q, e)
    print(reasoner.bn.get_all_variables())


def test_map():
    reasoner = BNReasoner('testing/lecture_example2.BIFXML')
    Q = {'I', 'J'}
    e = pd.Series({'O': True})
    assignments = reasoner.map(Q, e)
    print(assignments)


def test_mpe():
    reasoner = BNReasoner('testing/lecture_example.BIFXML')

    e = pd.Series({'Winter?': True})
    assignments = reasoner.mpe(e)
    print(assignments)


def test_dsep():
    reasoner = BNReasoner('testing/lecture_example.BIFXML')
    print(reasoner.dsep({"Rain?"}, {"Wet Grass?"}, {"Winter?"}))


def test_independence():
    reasoner = BNReasoner('testing/lecture_example.BIFXML')
    print(reasoner.dsep({"Winter?"}, {"Wet Grass?"}, {"Slippery Road?"}))


def test_marginalization():
    reasoner = BNReasoner('testing/lecture_example.BIFXML')
    print(reasoner.marginalize("Wet Grass?", "Rain?"))


def test_maxingout():
    reasoner = BNReasoner('testing/lecture_example.BIFXML')
    print(reasoner.maxing_out("Wet Grass?", "Rain?"))


def test_factor_multiplication():
    reasoner = BNReasoner('testing/lecture_example.BIFXML')
    print(reasoner.factor_mult("Wet Grass?", "Slippery Road?"))


def test_min_fill():
    reasoner = BNReasoner('testing/lecture_example.BIFXML')
    print(reasoner.minfill_ordering(["Rain?", "Winter?", "Sprinkler?"]))


def test_min_degree():
    reasoner = BNReasoner('testing/lecture_example.BIFXML')
    print("---\n", reasoner.bn.get_all_cpts())  # remove
    print(reasoner.min_degree_ordering(["Rain?", "Winter?", "Sprinkler?", "Slippery Road?", 'Wet Grass?']))


def test_variable_elimination():
    reasoner2 = BNReasoner('testing/lecture_example.BIFXML')
    print(reasoner2.bn.get_all_variables())
    print(reasoner2.variable_elimination2(["Rain?", "Winter?", "Sprinkler?"]))


def main():
    test_dsep()
    test_independence()
    test_prune()
    test_map()
    test_marginalization()
    test_maxingout()
    test_factor_multiplication()
    test_min_fill()
    test_min_degree()
    test_variable_elimination()
    test_map()
    test_mpe()


if __name__ == '__main__':
    main()
