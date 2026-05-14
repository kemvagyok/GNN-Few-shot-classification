from training.trainer import Trainer
from training.trainerDDP import TrainerDDP

from training.trainerOnlyEmbedding import TrainerOnlyEmbedding
from training.trainerOnlyEmbedding_DDP import TrainerOnlyEmbedding_DDP
#----------------------------------------------------

from utils import graph_builderFAISS, graph_builderFAISS_withGPU, graph_builderTorch, graph_builderFAISS_withGPUProba

#----------------------------------------------------

def get_trainer(is_ddp, embedder, gnn, criterion, metrics, config, device, local_rank):
    
    if config.graph_builder == "faiss":
        print("Graphbuilder type: FAISS")
        graph_builder = graph_builderFAISS
    elif config.graph_builder == "faiss_gpu":
        print("Graphbuilder type: FAISS with GPU")
        graph_builder = graph_builderFAISS_withGPU
    elif config.graph_builder == "faiss_gpu_proba":
        print("Graphbuilder type: FAISS with GPUProba")
        graph_builder = graph_builderFAISS_withGPUProba
    elif config.graph_builder == "pytorch":
        print("Graphbuilder type: PyTorch")
        graph_builder = graph_builderTorch

    if config.train_mode == "full":
        if not is_ddp:
            print("Trainer type: Embedding-> Full, GNN -> Minibatch, SINGLE")
            return Trainer(
            embedder=embedder,
            gnn=gnn,
            graph_builder=graph_builder,
            criterion=criterion,
            config=config,
            device=device,
            metric_fn=metrics
        )
        else:
            print("Trainer type: Embedding-> Full, GNN -> Minibatch, DDP")
            return TrainerDDP(
                embedder=embedder,
                gnn=gnn,
                graph_builder=graph_builder,
                criterion=criterion,
                config=config,
                local_rank=local_rank,
                metric_fn=metrics,
                device=device
            )
    else:
        if not is_ddp:
            print("Trainer type: Only Embedding-> Full, SINGLE")
            return TrainerEmbeddingOnlyMinibatch(
                embedder= embedder,
                criterion = criterion,
                config = config,
                device = device,
                metric_fn = metrics
            )
        else:
            print("Trainer type: Only Embedding-> Full, DDP")
            return TrainerEmbeddingOnlyMinibatch_DDP(
                embedder = embedder,
                criterion = criterion,
                config = config,
                rank = local_rank,
                metric_fn = metrics
            )