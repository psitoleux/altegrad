# altegrad
This is the repository of the team PCD (Paul Sitoleux, Clément Teulier, Damien Vilcocq) for the 2023 ALTeGraD data challenge.

Our experiments can be run from standard Google Colab/Kaggle notebooks, with two additional dependencies 

```bash
pip install info_nce
pip install torch_geometric
```

Then, after cloning this repository and having set the corresponding directory as the working one, using 

```python
%run main.py --epochs 32 --batch_size 80 --lr 5e-5 --trainable all_but_embeddings --Tmin 0.05 --Tmax 0.2 --epochs_per_cycle 6 --graph_encoder graph_transformer
```

will train a model reaching ~ 0.885 public score (training on a P100, taking approximately 7 hours).
