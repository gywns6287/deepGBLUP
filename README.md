
# deepGBLUP: Integration of deep learning and GBLUP for accurate genomic prediction.
 

## Model summary
We propose a novel genomic prediction algorithm, which integrates deep learning and GBLUP (deepGBLUP). Given SNP markers, the proposed deepGBLUP first extracts epistasis based on locally-connected layers. Then, it estimates initial breeding values through a relation-aware module, which extends GBLUP to the deep learning framework. Finally, deepGBLUP estimates a breeding value bias using a fully connected layer and adds it to the initial breeding value for calculating the final breeding values.

![image](https://user-images.githubusercontent.com/71325306/208086734-9aae626f-d672-4f9c-8cea-c222e926a9f8.png)
