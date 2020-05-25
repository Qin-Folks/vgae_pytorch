import torch


def get_gram_graph_embedding(node_attrs, adj, is_soft):
    node_num = node_attrs.shape[0]
    assert node_num <= adj.shape[0]

    if is_soft:
        adj = adj / 0.5
        adj_hard = (adj > 0.5).int()
        adj = ((adj_hard - adj).detach() + adj)[:node_num, :node_num]

    adj = adj[:node_num, :node_num]
    walk = node_attrs
    v1 = torch.sum(walk, dim=1)

    walk = torch.mm(adj, walk) * node_attrs
    v2 = torch.sum(walk, dim=1)

    walk = torch.mm(adj, walk) * node_attrs
    v3 = torch.sum(walk, dim=1)

    # walk = torch.mm(adj, walk) * node_attrs
    # v4 = torch.sum(walk, dim=1)
    #
    # walk = torch.mm(adj, walk) * node_attrs
    # v5 = torch.sum(walk, dim=1)
    #
    # walk = torch.mm(adj, walk) * node_attrs
    # v6 = torch.sum(walk, dim=1)

    # embedded_graph_matrix = torch.stack([v1, v2, v3, v4, v5, v6], dim=1)
    embedded_graph_matrix = torch.stack([v1, v2, v3], dim=1)
    return embedded_graph_matrix