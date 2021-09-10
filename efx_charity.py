import numpy as np
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
from networkx.exception import NetworkXNoCycle
from networkx.algorithms.dag import is_directed_acyclic_graph
from networkx.algorithms.dag import topological_sort
import sys

# rows are comma-separated
def row_inp_to_array(inp):
    return np.array([float(i) for i in inp.split(',')])

def setup():
    n = int(input('number of players: '))
    assert n > 0, 'must be a positive integer'

    t = int(input('number of item types: '))
    assert t > 0,  'must be a positive integer'

    items = row_inp_to_array(input('number of items of each type (comma-separated): '))
    assert len(items) == t, 'must be of length t=' + str(t)

    valuations = np.zeros((n,t))
    for i in range(n):
        val_row = row_inp_to_array(input('valuation for player ' + str(i) + ' (comma-separated): '))
        assert len(val_row) == t, 'must be of length t=' + str(t)
        valuations[i,:] = val_row

    assignments = np.zeros((n,t))
    return n, t, assignments, items, valuations

#envy-freeness
def get_bundle_values(assignments, valuations, j):
    return assignments.dot(valuations[j])

def check_ef(assignments, valuations, j):
    bundle_values = get_bundle_values(assignments, valuations, j)
    return j in np.flatnonzero(bundle_values == np.max(bundle_values))

def check_efx(assignments, valuations, j):
    if check_ef(assignments, valuations, j):
        return True
    bundle_values = get_bundle_values(assignments, valuations, j)
    # for every item type:
    for i in range(len(assignments[0])):
        amt_of_type_i = np.copy(assignments[:, i])
        without_1 = np.copy(amt_of_type_i)

        # just consider nonzero bundles and j
        mask = list(np.flatnonzero(amt_of_type_i))
        if j not in mask:
            mask.append(j)

        # of nonzero bundles, remove an item
        without_1[mask] -= 1
        # dont subtract one from the current player
        without_1[j] = amt_of_type_i[j]
        # should be EF among nonzero items
        assignments[:, i] = without_1
        is_ef = check_ef(assignments[mask, :], valuations[mask, :], mask.index(j))
        assignments[:, i] = amt_of_type_i
        if not is_ef:
            return False
    return True

def check_all_ef(assignments, valuations):
    return check_all(assignments, valuations, check_ef)

def check_all_efx(assignments, valuations):
    return check_all(assignments, valuations, check_efx)

def check_all(assignments, valuations, fn):
    for j in range(len(assignments)):
        if not fn(assignments, valuations, j):
            return False
    return True

def check_all_done(assignments, items, strict=True):
    # no more items to assign
    if strict:
        return np.all(np.sum(assignments, axis=0) == items)
    return np.all(np.sum(assignments, axis=0) >= items)


def item_type_available(assignments, items, i):
    return np.sum(assignments[:, i]) < items[i]

def run_u0_if_possible(n, t, assignments, items, valuations):
    item_order = np.arange(t)
    np.random.shuffle(item_order)
    for i in item_order:
        if item_type_available(assignments, items, i):
            player_order = np.arange(n)
            np.random.shuffle(player_order)
            for j in player_order:
                assignments[j,i] += 1
                if check_all_efx(assignments, valuations):
                    return True, assignments
                assignments[j,i] -= 1
    return False, assignments

def create_envy_digraph(assignments, valuations, n):
    envy_edges = np.zeros((n, n))
    for i in range(n):
        row = assignments.dot(valuations[i])
        envy_edges[i, :] = row > row[i]
    return nx.DiGraph(envy_edges)

def locate_envy_cycle(G):
    cycles = list(simple_cycles(G))
    if len(cycles) == 0:
        return False, None
    np.random.shuffle(cycles)
    return True, cycles[0]

def make_self_envy_fn(old_assignment, valuation):
    return lambda new_assignment: new_assignment.dot(valuation) > old_assignment.dot(valuation)

def get_mea(s, assignments_inp, items, valuations, n):
    is_self_envy = make_self_envy_fn(assignments_inp[s,:], valuations[s])
    assignments = np.copy(assignments_inp)
    item_order = np.arange(len(assignments[0]))
    np.random.shuffle(item_order)
    for i in item_order:
        if item_type_available(assignments, items, i):
            # add type i item to source
            assignments[s,i] += 1

            # ensure there are enviers
            G = create_envy_digraph(assignments, valuations, n)
            assert get_first_ancestor(G, s) != None or is_self_envy(assignments[s,:])

            # now search for MEA
            for k in item_order:
                if k == i:
                    continue
                while assignments[s,k] > 0:
                    assignments[s,k] -= 1
                    # if no more enviers, then put it back and try a different one
                    G = create_envy_digraph(assignments, valuations, n)
                    if get_first_ancestor(G, s) == None and not is_self_envy(assignments[s,:]):
                        assignments[s,k] += 1
                        break

            # this should be a minimal envied subset
            for k in item_order:
                if assignments[s,k] == 0:
                    continue
                assignments[s,k] -= 1
                G = create_envy_digraph(assignments, valuations, n)
                assert get_first_ancestor(G, s) == None and not is_self_envy(assignments[s,:])
                assignments[s,k] += 1
            # choose first MEA here. (later, we can try every MEA....)
            G = create_envy_digraph(assignments, valuations, n)
            mea = get_first_ancestor(G, s)
            if mea == None:
                assert is_self_envy(assignments[s,:])
                mea = s
            return mea, i, assignments[s, :]

def from_t_i_to_s_i(G, node):
    path = [node]
    while True:
        node = get_first_ancestor(G, node)
        if node == None:
            break
        path.append(node)
    return path

def get_first_ancestor(G, node):
    for k in G.predecessors(node):
        node = k
        return node
    return None

def run_u2_if_possible(n, assignments, items, valuations, pr):
    G = create_envy_digraph(assignments, valuations, n)
    all_sources = [s for s,d in G.in_degree() if d == 0]
    np.random.shuffle(all_sources)
    # choose a source s0
    for s in all_sources:
        pr("trying U2 with source:", s)
        success, assignments = try_u2_with_given_s0(n, assignments, items, valuations, s)
        if success:
            return success, assignments
    return False, assignments

def try_u2_with_given_s0(n, assignments_inp, items, valuations, s):
    assignments = np.copy(assignments_inp)
    G = create_envy_digraph(assignments, valuations, n)
    all_sources = [s for s,d in G.in_degree() if d == 0]
    # choose a source s0
    sources = [s]
    items_used = np.zeros(len(assignments[0]), dtype=int)
    zs = []
    t_i_list = []

    while True:
        # if no more items, then can't succeed
        items -= items_used
        no_more_items = check_all_done(assignments, items, strict=False)
        items += items_used
        if no_more_items:
            return False, assignments

        # get MEA and minimal subset
        t_i, item_type_used, z_iminus1 = get_mea(sources[-1], assignments, items, valuations, n)
        items_used[item_type_used] += 1
        zs.append(z_iminus1)
        t_i_list.append(t_i)

        # get all sources
        accessible_sources = []
        for s in all_sources:
            if s == t_i:
                accessible_sources.append(s)
                continue
            for path in nx.all_simple_paths(G, source=s, target=t_i):
                accessible_sources.append(s)
                break
        # if we've seen any of them before, then we found a loop
        overlapping_sources = set(sources).intersection(accessible_sources)
        if len(overlapping_sources) > 0:
            break
        sources.append(accessible_sources[0])

    # re-assign items
    repeated_source = next(iter(set(overlapping_sources)))
    source_sighting = sources.index(repeated_source)
    sources.append(repeated_source)
    # we start from after the source sighting until the end
    # reassign paths first
    s_and_t_and_z = zip(sources[source_sighting+1:], t_i_list[source_sighting:], zs[source_sighting:])
    for s, t, _ in s_and_t_and_z:
        for path in nx.all_simple_paths(G, source=s, target=t):
            assignments[path[:-1], :] = assignments[path[1:], :]
            break
    # then assign zs
    s_and_t_and_z = zip(sources[source_sighting+1:], t_i_list[source_sighting:], zs[source_sighting:])
    for _, t, z in s_and_t_and_z:
        assignments[t, :] = z
    return True, assignments

def draw_envy(assignments, valuations, n):
    G = create_envy_digraph(assignments, valuations, n)
    nx.draw(G, with_labels=True)

def run(inputs=None, log=True):
    pr = print if log else (lambda *x: None)
    # setup
    n, t, assignments, items, valuations = inputs if inputs else setup()
    # algorithm
    while True:
        # assert partial EFX
        assert check_all_efx(assignments, valuations), "this should be partial EFX"
        # de cycle graph if needed
        G = create_envy_digraph(assignments, valuations, n)
        if not is_directed_acyclic_graph(G):
            is_cycle, cycle = locate_envy_cycle(G)
            while is_cycle:
                pr("DE-CYCLING GRAPH")
                cycle_mask = cycle
                cycle_mask_new = cycle_mask[1:] + cycle_mask[0:1]
                assignments[cycle_mask_new] = assignments[cycle_mask]
                G = create_envy_digraph(assignments, valuations, n)
                is_cycle, cycle = locate_envy_cycle(G)
        assert is_directed_acyclic_graph(G)
        # check if done
        if check_all_done(assignments, items):
            # print("DONE")
            return n, t, assignments, items, valuations
        # run U0 if possible
        success, assignments = run_u0_if_possible(n, t, assignments, items, valuations)
        if success:
            pr("APPLIED U0")
            continue
        pr("COULD NOT APPLY U0")
        # run U2 if possible
        success, assignments = run_u2_if_possible(n, assignments, items, valuations, pr)
        if success:
            pr("APPLIED U2")
            continue
        print("COULD NOT APPLY U2 (!)")
        return n, t, assignments, items, valuations

def run_and_check(draw=False, **kwargs):
    n, t, assignments, items, valuations = run(**kwargs)
    for idx,row in enumerate(assignments):
        print('player', idx, 'assignments:', row)
    assert check_all_efx(assignments, valuations), "EFX SHOULD HOLD"
    assert check_all_done(assignments, items), "ALL ITEMS SHOULD BE ALLOCATED"
    if draw:
        draw_envy(assignments, valuations, n)
    return n, t, assignments, items, valuations

print("loaded efx algorithm")

if len(sys.argv) > 1 and sys.argv[1] == 'run':
    run_and_check()