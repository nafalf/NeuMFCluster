import torch
import torch.nn as nn

class NeuMFCluster(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        n_clusters,
        emb_dim=64,
        cluster_dim=16,
        dropout=0.2,
        mlp_layer=None
    ):
        super().__init__()
        
        if mlp_layer is None:
            mlp_layer = [128, 64, 32]

        self.user_gmf = nn.Embedding(n_users, emb_dim)
        self.item_gmf = nn.Embedding(n_items, emb_dim)

        self.user_mlp = nn.Embedding(n_users, emb_dim)
        self.item_mlp = nn.Embedding(n_items, emb_dim)
        self.cluster_emb = nn.Embedding(n_clusters, cluster_dim)

        mlp_input_dim = emb_dim * 2 + cluster_dim

        # Membangun MLP secara dinamis
        mlp_modules = []
        current_dim = mlp_input_dim
        for hidden_dim in mlp_layer:
            mlp_modules.append(nn.Linear(current_dim, hidden_dim))
            mlp_modules.append(nn.Dropout(dropout))
            mlp_modules.append(nn.ReLU())
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_modules)

        # Layer output: GMF (emb_dim) + output layer terakhir dari MLP
        self.output = nn.Linear(emb_dim + current_dim, 1)

    def forward(self, user, item, cluster):
        gmf = self.user_gmf(user) * self.item_gmf(item)

        mlp_in = torch.cat(
            [
                self.user_mlp(user),
                self.item_mlp(item),
                self.cluster_emb(cluster)
            ],
            dim=-1
        )

        x = torch.cat([gmf, self.mlp(mlp_in)], dim=-1)
        return self.output(x).squeeze(-1)


class NeuMFBaseline(nn.Module):
    def __init__(
        self, 
        n_users, 
        n_items, 
        emb_dim=64, 
        dropout=0.2,
        mlp_layer=None
    ):
        super().__init__()
        
        if mlp_layer is None:
            mlp_layer = [64, 32]

        self.user_gmf = nn.Embedding(n_users, emb_dim)
        self.item_gmf = nn.Embedding(n_items, emb_dim)

        self.user_mlp = nn.Embedding(n_users, emb_dim)
        self.item_mlp = nn.Embedding(n_items, emb_dim)

        mlp_input_dim = emb_dim * 2

        mlp_modules = []
        current_dim = mlp_input_dim
        for hidden_dim in mlp_layer:
            mlp_modules.append(nn.Linear(current_dim, hidden_dim))
            mlp_modules.append(nn.Dropout(dropout))
            mlp_modules.append(nn.ReLU())
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_modules)

        self.output = nn.Linear(emb_dim + current_dim, 1)

    def forward(self, user, item):
        gmf = self.user_gmf(user) * self.item_gmf(item)
        mlp_in = torch.cat([self.user_mlp(user), self.item_mlp(item)], dim=-1)
        x = torch.cat([gmf, self.mlp(mlp_in)], dim=-1)
        return self.output(x).squeeze(-1)
