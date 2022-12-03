import torch


def fully_connect(num_patches, width, length):

    return torch.ones((num_patches, num_patches))


def direct_one_hop(num_patches, grid_size):

    width, length = grid_size
    adj = torch.zeros([num_patches, num_patches])

    for node in range(num_patches):

        res = node % length

        if node - length > 0:
            adj[node, node - length] = 1
            adj[node - length, node] = 1
        
        if node + length < num_patches - 1:
            adj[node, node + length] = 1
            adj[node + length, node] = 1

        if res - 1 >= 0:
            adj[node, node - 1] = 1
            adj[node - 1, node] = 1
        
        if res + 1 < length:
            adj[node, node + 1] = 1
            adj[node + 1, node] = 1

        adj[node, node] = 1

    return adj

def all_one_hop(num_patches, grid_size):

    width, length = grid_size
    adj = torch.zeros([num_patches, num_patches])

    for node in range(num_patches):

        if node - length > 0:
            adj[node, node - length] = 1
            adj[node - length, node] = 1

            if (node - length) % length - 1 >= 0:
                adj[node, node - length - 1] = 1
                adj[node - length - 1, node] = 1

            if (node - length) % length + 1 < length:
                adj[node, node - length + 1] = 1
                adj[node - length + 1, node] = 1
        
        if node + length < num_patches - 1:
            adj[node, node + length] = 1
            adj[node + length, node] = 1

            if (node + length) % length - 1 >= 0:
                adj[node, node + length - 1] = 1
                adj[node + length - 1, node] = 1
            
            if (node + length) % length + 1 < length:
                adj[node, node + length + 1] = 1
                adj[node + length + 1, node] = 1

        res = node % length

        if res - 1 >= 0:
            adj[node, node - 1] = 1
            adj[node - 1, node] = 1
        
        if res + 1 < length:
            adj[node, node + 1] = 1
            adj[node + 1, node] = 1

        adj[node, node] = 1

    return adj

