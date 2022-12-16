
# deepGBLUP: Integration of deep learning and GBLUP for accurate genomic prediction.
 

## Model summary
We propose a novel genomic prediction algorithm, which integrates deep learning and GBLUP (deepGBLUP). Given SNP markers, the proposed deepGBLUP first extracts epistasis based on locally-connected layers. Then, it estimates initial breeding values through a relation-aware module, which extends GBLUP to the deep learning framework. Finally, deepGBLUP estimates a breeding value bias using a fully connected layer and adds it to the initial breeding value for calculating the final breeding values.

<img src = "https://user-images.githubusercontent.com/71325306/208086095-3471a61a-baf3-4db0-8a42-18f81ebe5842.png" width="50%" height="50%">
