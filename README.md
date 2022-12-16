
# deepGBLUP: Integration of deep learning and GBLUP for accurate genomic prediction.
 

## Model summary
We propose a novel genomic prediction algorithm, which integrates deep learning and GBLUP (deepGBLUP). Given SNP markers, the proposed deepGBLUP first extracts epistasis based on locally-connected layers. Then, it estimates initial breeding values through a relation-aware module, which extends GBLUP to the deep learning framework. Finally, deepGBLUP estimates a breeding value bias using a fully connected layer and adds it to the initial breeding value for calculating the final breeding values.

![model](https://user-images.githubusercontent.com/71325306/208086295-745f0917-3d00-42c2-8bda-1f9ca30a3661.png)
