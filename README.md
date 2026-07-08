# Transfer Learning for Multi-label Medical Image Classification

Deep Learning Final Project — University of Oulu, Fall 2025

## Overview

Multi-label classification of retinal diseases from fundus images using transfer learning on the **ODIR** (Ocular Disease Intelligent Recognition) dataset. Three conditions are predicted simultaneously per image:

- **DR** — Diabetic Retinopathy
- - **G** — Glaucoma
  - - **AMD** — Age-related Macular Degeneration
   
    - ## Approach
   
    - The project is structured as four progressive stages, each building on the best model found in the previous one:
   
    - 1. **Transfer learning strategies** — ResNet18 vs. EfficientNet, compared with no fine-tuning, a frozen backbone, and full fine-tuning
      2. 2. **Loss functions for class imbalance** — Focal Loss vs. Class-Balanced Loss
         3. 3. **Attention mechanisms** — Squeeze-and-Excitation (SE) and Multi-Head Attention (MHA) added on top of ResNet18
            4. 4. **Ensemble methods** — simple voting, weighted averaging, and a selective top-2 ensemble combining the best models from stages 1–3
              
               5. ## Results (offsite test set, average F-score across DR / G / AMD)
              
               6. | Stage | Model | F-score |
               7. |---|---|---|
               8. | 1.1 No fine-tuning | ResNet18 | 51.26% |
               9. | 1.1 No fine-tuning | EfficientNet | 55.41% |
               10. | 1.2 Frozen backbone | ResNet18 | 45.31% |
               11. | 1.2 Frozen backbone | EfficientNet | 70.74% |
               12. | 1.3 Full fine-tuning | ResNet18 | 78.92% |
               13. | 1.3 Full fine-tuning | EfficientNet | 69.34% |
               14. | 2. Focal Loss | ResNet18 | 75.84% |
               15. | 2. Class-Balanced Loss | ResNet18 | 77.82% |
               16. | 3. SE Attention | ResNet18 | 77.01% |
               17. | 3. MHA Attention | ResNet18 | 80.53% |
               18. | 4. Ensemble — Simple Voting (4 models) | — | 78.07% |
               19. | 4. Ensemble — Weighted Average (4 models) | — | 78.79% |
               20. | **4. Selective Ensemble — Top 2 (MHA + Full FT)** | — | **81.57%** |
              
               21. **Best single model:** ResNet18 + Multi-Head Attention (80.53%)
               22. **Best overall:** Selective top-2 ensemble (ResNet18 MHA + ResNet18 Full FT), weighted by each model's validation score — a +1.05% improvement over the best individual model.
              
               23. ## Key findings
              
               24. - Fine-tuning strategy matters more than backbone choice: a frozen ResNet18 performs worse (45%) than a plain pretrained EfficientNet with no fine-tuning at all (55%), but full fine-tuning brings ResNet18 to the top individual score.
                   - - Multi-Head Attention gave the largest single-model gain, improving on plain full fine-tuning by ~1.6 points.
                     - - Combining all four trained models blindly (voting or weighted average) actually **hurt** performance versus the single best model — weaker models diluted the ensemble. Selectively combining only the top 2 was what delivered a real improvement, showing that ensemble diversity isn't automatically beneficial when models are correlated.
                      
                       - ## Tech stack
                      
                       - `PyTorch` · `torchvision` · Google Colab (T4 GPU)
                      
                       - ## Repository structure
                      
                       - ```
                         notebooks/   → full experiment pipeline (Tasks 1–4): data exploration, training, evaluation, ensembling
                         results/     → prediction exports
                         src/         → reserved for reusable modules
                         ```
                         
