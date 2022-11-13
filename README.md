# NBFNet: Neural Bellman-Ford Networks #

[Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction][paper]

[Zhaocheng Zhu](https://kiddozhu.github.io),
[Zuobai Zhang](https://oxer11.github.io),
[Louis-Pascal Xhonneux](https://github.com/lpxhonneux),
[Jian Tang](https://jian-tang.com)

[paper]: https://arxiv.org/pdf/2106.06935.pdf

## Reproduction ##

We provide the hyperparameters for each experiment in configuration files.
All the configuration files can be found in `config/*/*.yaml`.

To reproduce the results of NBFNet, use the following command.

train
```bash
python -m torch.distributed.launch --nproc_per_node 4 --master_port 25211 kddcup_main.py -c config/knowledge_graph/bellmanford_kddcup_best.yaml
```

valid

```bash
python -m torch.distributed.launch --nproc_per_node 4 --master_port 25211 kddcup_main.py -c config/knowledge_graph/bellmanford_kddcup_valid.yaml
```

test_dev, test_challenge

```bash
python -m torch.distributed.launch --nproc_per_node 4 --master_port 25211 kddcup_main.py -c config/knowledge_graph/bellmanford_kddcup_test.yaml
```
The config file of valid and test add checkpoint and test parameters on the basis of the train config file

The file path of the checkpoint and predicted score is in the output_dir