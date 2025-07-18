---
layout: project_page
permalink: /

title: Fashion recommendation system based on personal characteristics
authors:
    Seonghoon Lee, Taehun Kim
affiliations:
    Electrical and Computer Engineering, Pusan National University
paper: https://github.com/pnucse-capstone-2024/Capstone-2024-team-24/blob/main/docs/01.%EB%B3%B4%EA%B3%A0%EC%84%9C/03.%EC%B5%9C%EC%A2%85%EB%B3%B4%EA%B3%A0%EC%84%9C.pdf
video: https://youtu.be/bAQ_bLVpX28?feature=shared
code: https://github.com/pnucse-capstone-2024/Capstone-2024-team-24
---

<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
This research focuses on developing a personalized fashion recommendation system based on individual physical characteristics. In modern society, fashion serves as a crucial means of self-expression, yet finding suitable styles that complement one's physical attributes remains challenging. The system analyzes three key physical characteristics-personal color, face shape, and body type-to provide customized fashion recommendations.
The methodology includes image analysis using deep learning techniques to automatically extract users' physical characteristics from uploaded photos. For personal color determination, the system analyzes skin color values using K-means clustering to classify users into four seasonal types (Spring, Summer, Autumn, Winter). Face shape classification employs an EfficientNetB4-based model trained on a dataset of 5,000 images, achieving 75.83% test accuracy in identifying five face shapes. Body type measurement utilizes MediaPipe's Pose Landmark feature to extract key points and calculate proportions between different body parts.
Two recommendation approaches were implemented: a Random Forest-based system that predicts ratings based on clothing attributes and extracted physical characteristics (achieving test RMSE of 0.759 for men and 0.773 for women), and a content-based collaborative filtering system using cosine similarity to recommend similar items based on keyword of clothes. The Random Forest-based system also incorporates user feedback to continuously improve recommendation quality.
The research demonstrates the feasibility and effectiveness of personalized fashion recommendations based on physical characteristics, contributing to the growing demand for individualized services. Future work includes enhancing the system with more diverse data, real-time feedback incorporation, and integration with e-commerce platforms.
        </div>
    </div>
</div>


# Overview
This technical report presents a comprehensive fashion recommendation system that analyzes individual physical characteristics—personal color, face shape, and body type—to provide customized fashion suggestions. The system addresses the growing demand for personalized services in the fashion industry.
# Key Components
## Personal Color Analysis
The system extract RGB values of skin area of a face image, and convert it to HSV values and Lab color space. The system uses K-means clustering to classify users' skin tones into four seasonal types (Spring, Summer, Autumn, Winter) by analyzing HSV values and Lab color space. This classification helps recommend colors that enhance the user's natural complexion.
## Face Shape Recognition
We use an EfficientNetB4-based deep learning model. The model was built based on the EfficientNetB4 network pretrained on the ImageNet dataset. A GlobalAveragePooling2D layer was applied to the output of EfficientNetB4, and the resulting output was passed through a Fully-connected layer to classify into five face shapes. We initially trained the model with the EfficientNetB4 backbone frozen for 25 epochs, learning rate 1e-3 and kernel regularizer l2(1e-3), and then fine-tuned the entire network by unfreezing all layers and using a low learning rate(1e-4) with 20 epochs. We use EarlyStopping(monitor='val_accuracy', patience=10,restore_best_weights=True) callbacks. The system identifies five face shapes (heart, long, oval, round, square) with 75.8% accuracy. These classifications inform recommendations for clothing styles and necklines that complement facial features.
## Body Shape Measurement
The system employs MediaPipe's Pose Landmark technology and Rembg's removing background technology to extract key body measurements and proportions. It calculates ratios between shoulders, waist, hips, and chest to classify users into different body types, enabling recommendations that enhance body proportions.
## Recommendation Engines
Two approaches were implemented:

### A Random Forest model
This model predicts ratings based on user characteristics and clothing attributes(appropriate wearing situation, fit, color, mood, style, season). We trained the model with [fashion preferences and recommendation data by year](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71446). This model has test RMSE scores of 0.759 for men and 0.773 for women. It also inorporates user feedback to continuously improve recommendation quality. The user feedback system was implemented by adding the  (user characteristic - clothes attributes - rating) data to the existing training data and learning a new Random Forest model when the user gave a rating.

### A content-based collaborative filtering system 
This model uses cosine similarity to find similar clothing items based on keyword of the clothes. We use TF-IDF Vectorizing to embedding the keyword. We collected (clothes - keyword) data from [K-Fashion](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=51) data. We assign fashion keywords based on the physical characteristics and the model recommend clothing items that share similar keywords. Alternatively, the model recommend items that match the keywords entered by the user.


# Conclusion and Applications
The possibility of personalized services was confirmed by establishing a system for recommending customized fashion based on the physical characteristics of users such as personal color, face type, and body type. In particular, the Random Forest-based recommendation system recorded high prediction accuracy, and the recommended quality improvement function reflecting user feedback played an important role in continuously improving the performance of the system.
These results showed the importance of customized services and the possibility of how the personalized recommendation system can be applied in the fashion industry.

This technology enables highly personalized fashion recommendations that consider individual physical attributes rather than just following general trends. The web interface allows users to upload photos or manually input their characteristics to receive tailored fashion suggestions.
# Future Directions
Future enhancements include expanding the dataset, incorporating real-time trend analysis, and potential integration with e-commerce platforms to create a seamless shopping experience.

---

# Data
- [fashion preferences and recommendation data by year](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71446)
    - This data is the outcome of a project by the National Information Society Agency (NIA) of Korea.
- [K-Fashion](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=51)
    - This data is the outcome of a project by the National Information Society Agency (NIA) of Korea.
- [Face Shape dataset](https://www.kaggle.com/datasets/niten19/face-shape-dataset)
- [Human Body classification dataset](https://doi.org/10.25405/data.ncl.19307300.v1)

- [Korean bodyshape data](https://sizekorea.kr/human-info/body-shape-class/age-gender-body)
    - This data is the outcome of a project by Korean Agency for Technology and Standards (KATS), Ministry of Trade, Industry and Energy (MOTIE) of Korea.

# Tables

|               | man dataset | woman dataset |
|---------------|-------------|---------------|
| Train dataset | 0.304       | 0.312         |
| Test dataset  | 0.759       | 0.773         |

*Table 9: Root Means Squared Error(RMSE) of Random Forest-based recommendation system.*

# Figures
![figure1](/static/image/figure1.PNG)

*Figure 1: System Architecture*

![figure2](/static/image/figure2.PNG)

*Figure 2: estimated measurements and locations for bust, waist, shoulder, and hips using MediaPipe. The dataset is (6)*

![figure3](/static/image/figure3.PNG)

*Figure 3: The average color values for each personal color season (Spring, Summer, Autumn, Winter) identified through K-means clustering. The three parameters shown are Value (brightness), Saturation (color intensity), and b channel (yellow-blue axis). Spring shows high brightness and moderate intensity with warm undertones. Summer exhibits high brightness but low intensity with cool undertones. Autumn features lower brightness with warm undertones. Winter combines moderate brightness with higher intensity and cooler undertones.*

![figure4](/static/image/figure4.PNG)

*Figure 4: Learning graph of Faceshape classification model based on EfficientNetB4. We first trained the model 25 epoches with freezing the weights of pretrained EfficientNetB4. At the final epoch, training accuracy was about 98.51 percent and test accuracy was about 69.61 percent. This model classify the faceshape(Heart, Oblong, Oval, Round, Square) from single face image*

![figure5](/static/image/figure5.PNG)

*Figure 5: Learning graph of Faceshape classification model based on EfficientNetB4 without freezing the pretrained model. We trained the model 20 epoches. Training accuracy was about 99.93 percent and test accuracy was about 75.83 percent at the best weights*

![figure6](/static/image/figure6.PNG)

*Figure 6. Fashion recommend system webpage*

[![figure7.png](https://i.postimg.cc/FFq2XM9X/figure7.png)](https://postimg.cc/kDQjxh6T)

*Figure 7. Fashion recommendation result*

# Acknowledgements
This graduation project was conducted as part of the 2024 Early-Year Capstone Design Program of the School of Information and Computer Engineering. It was carried out as an industry-academic collaboration project, with partial guidance and advice provided by an employee of NAVER Webtoon. This work has been registered with the Korea Copyright Commission (Registration No. C-2024-042740), with  Research and Business Development Foundation of Pusan National University listed as the copyright holder. The author participated in the work as the creators.

# References
1. Yun-Seok Jung, “A Study on the Quantitative Diagnosis Model of Personal 
Color,” Journal of Convergence for Information Technology, Vol. 11, No. 11, pp. 
277-287, 2021. (in Korean) doi: [https://doi.org/10.22156/CS4SMB.2021.11.11.277](https://doi.org/10.22156/CS4SMB.2021.11.11.277)
2. Jong-Suk An, “A Study on Effective Image Making Depending on Hair Style 
and Neckline,” Jounal of The Korean Society of cosmetology, Vol. 15, No. 1, pp.342-351, 2009. (in 
Korean) Available at [https://www.riss.kr/link?id=A76494548](https://www.riss.kr/link?id=A76494548)
3. Soo-ae Kwon, Fashion and Life, Gyohakyungusa, 2016. ISBN: 9788935405558
4. “2023 Consumption Trend Series - 03 Personalized Services,” MezzoMedia 
Available: [https://www.mezzomedia.co.kr/data/insight_m_file/insight_m_file_1605.pdf](https://www.mezzomedia.co.kr/data/insight_m_file/insight_m_file_1605.pdf)
 (downloaded 2024, May. 19)
5. So-young Lee, “Personal Color Tone Type and Categorization of Harmonious 
Colors According to Skin Color,” M.S. thesis, Graduate School of Cultural and 
Information Policy, Hongik Univ., Seoul, South Korea, 2019. (in Korean) doi: [https://www.doi.org/10.23174/hongik.000000024122.11064.0000288](https://www.doi.org/10.23174/hongik.000000024122.11064.0000288)
6. Trotter, Cameron Patrick; Peleja, Filipa; Santos, Alberto de; Dotti, Dario (2023). Human Body Shape Classification Dataset. Newcastle University. Dataset. doi: [https://doi.org/10.25405/data.ncl.19307300.v1](https://doi.org/10.25405/data.ncl.19307300.v1)

# Appendix(Follow-up study)
After the capstone project is finished, we evaluate the recommendation system more precisely

We considered a score of 3 or higher to be preferred (or recommended) and a score of less than 3 to be disliked (or not recommended) based on the score the user gave to the clothes and the score predicted by the recommendation system.

Additionally, We reduce the face shape classification model size using EfficientNetB2

We upload the code for model test and face shape classification model on the Github: [Link](https://github.com/minchoCoin/capstone-team-24-page/tree/main/appendix)

## Precision and recall

We measured the precision and recall using classification_report function in sklearn.metrics.

|                    | Real positive | Real negative |
|--------------------|---------------|---------------|
| Predicted positive |       TP      |       FP      |
| Predicted negative |       FN      |       TN      |

(Table 10. precision and recall)

$$ precision = \frac{TP}{TP+FP}$$

$$ recall = \frac{TP}{TP+FN}$$

$$ accuracy = \frac{TP+TN}{TP+FP+FN+TN}$$


### Man

Figure 8 and Figure 9 shows the precision and recall of the recommendation system on the man dataset. 0 means not recommended or disliked, and 1 means recommended or preferred. These result show that the recommendation system has a high precision, which means that many of the recommended ones are preferred by the user

![fig8](/static/image/figure8.PNG)

*Figure 8. precision and recall of the recommendation system on the man training dataset. 0 means not recommended or disliked, and 1 means recommended or preferred*

![fig9](/static/image/figure9.PNG)

*Figure 9. precision and recall of the recommendation system on the man test dataset. 0 means not recommended or disliked, and 1 means recommended or preferred*

### Woman

Figure 10 and Figure 11 shows the precision and recall of the recommendation system on the woman dataset. 0 means not recommended or disliked, and 1 means recommended or preferred. These result show that the recommendation system has a high precision, which means that many of the recommended ones are preferred by the user

![fig10](/static/image/figure10.PNG)

*Figure 10. precision and recall of the recommendation system on the woman training dataset. 0 means not recommended or disliked, and 1 means recommended or preferred*

![fig11](/static/image/figure11.PNG)

*Figure 11. precision and recall of the recommendation system on the woman test dataset. 0 means not recommended or disliked, and 1 means recommended or preferred*

## Precision at k and recall at k

We measured the precision at k and recall at k using users with two or more ratings. precision at Top k measures the proportion of relevant items in the top K results returned by a system. recall at Top k measures the proportion of all relevant items that are included in the top K results.

$$ Precision @k = \frac{\text{Number of items relevant to the user in top K}}{K} $$


$$ Recall @ k = \frac{\text{Number of items relevant to the user in top K}}{\text{Total number of items relevant to the user}} $$
### Man

Figure 12 and Figure 13 shows the precision at Top k and recall at Top k. Figure 14 and 15 shows the precision and recall curve

![fig12](/static/image/figure12.png)

*Figure 12. precision at Top k and recall at Top k of the recommendation system on the man training dataset.*

![fig13](/static/image/figure13.png)

*Figure 13. precision at Top k and recall at Top k of the recommendation system on the man test dataset.*

![fig14](/static/image/figure14.png)
![fig14](/static/image/figure14-1.png)

*Figure 14. precision-recall curve of man training set and test set*

![fig15](/static/image/figure15.png)

*Figure 15. comparision of precision-recall curve of man training set and test set*

### Woman

Figure 16 and Figure 17 shows the precision at Top k and recall at Top k. Figure 18, 19 and 20 shows the precision and recall curve

![fig16](/static/image/figure16.PNG)

*Figure 16. precision at Top k and recall at Top k of the recommendation system on the woman training dataset.*

![fig17](/static/image/figure17.png)

*Figure 17. precision at Top k and recall at Top k of the recommendation system on the woman test dataset.*

![fig18](/static/image/figure18.png)

*Figure 18. precision-recall curve of man training set*

![fig19](/static/image/figure19.png)

*Figure 19. precision-recall curve of man test set*

![fig20](/static/image/figure20.png)

*Figure 20. comparision of precision-recall curve of woman training set and test set*

## Precision at k and recall at k computed based on the items relevant to the personal characteristic element

In this section, We measure precision at k and recall at k by comparing the actual preference of clothes by personal color, face type, and body type, and the clothes predicted by the recommendation system to be preferred for those characteristic

$$ Precision @ k = \frac{\text{Number of items relevant to the personal characteristic in top K}}{K}$$

$$ Recall @ k = \frac{\text{Number of items relevant to the personal characteristic in top K}}{\text{Total number of items relevant to the personal characteristic}}$$

### Man

Figure 21 and Figure 22 shows the precision at Top k and recall at Top k. Figure 23, 24 and 25 shows the precision and recall curve

![fig100](/static/image/figure100.png)

*Figure 21. precision at Top k and recall at Top k computed based on the items relevant to the personal characteristic element with the man training dataset.*

![fig101](/static/image/figure101.png)

*Figure 22. precision at Top k and recall at Top k computed based on the items relevant to the personal characteristic element with the man test dataset.*

![fig102](/static/image/figure102.png)

*Figure 23. Precision-recall curve computed based on the items relevant to the personal characteristic element with the man train dataset.*

![fig103](/static/image/figure103.png)

*Figure 24. Precision-recall curve computed based on the items relevant to the personal characteristic element with the man test dataset.*

![fig104](/static/image/figure104.png)

*Figure 25. Comparison of Precision-recall curve computed based on the items relevant to the personal characteristic element with the man test dataset.*

### Woman

Figure 26 and Figure 27 shows the precision at Top k and recall at Top k. Figure 28, 29 and 30 shows the precision and recall curve

![fig105](/static/image/figure105.png)

*Figure 26. precision at Top k and recall at Top k computed based on the items relevant to the personal characteristic element with the woman training dataset.*

![fig106](/static/image/figure106.png)

*Figure 27. precision at Top k and recall at Top k computed based on the items relevant to the personal characteristic element with the woman test dataset.*

![fig107](/static/image/figure107.png)

*Figure 28. Precision-recall curve computed based on the items relevant to the personal characteristic element with the woman train dataset.*

![fig108](/static/image/figure108.png)

*Figure 29. Precision-recall curve computed based on the items relevant to the personal characteristic element with the woman test dataset.*

![fig109](/static/image/figure109.png)

*Figure 30. Comparison of Precision-recall curve computed based on the items relevant to the personal characteristic element with the woman test dataset.*

## Limitation of measuring precision with test dataset

As shown in the Figure 32 and 34, in the test dataset, many of the users  rated only 1-2 clothes. This makes the precision and recall inaccurate. Therefore, the results of the test data above must be interpreted carefully.

![fig110](/static/image/figure110.png)

*Figure 31. Distribution of number of ratings per user in man training dataset*

![fig111](/static/image/figure111.png)

*Figure 32. Distribution of number of ratings per user in man test dataset*

![fig112](/static/image/figure112.png)

*Figure 33. Distribution of number of ratings per user in woman training dataset*

![fig113](/static/image/figure113.png)

*Figure 34. Distribution of number of ratings per user in woman test dataset*

## Model comparision
In this section, we compare the recommendation system based on xgboost, lightgbm, gradientboosting, randomforest, and decision tree model.

We created three groups of machine learning models with slightly different parameters

1. Group 1
```py
models = {
    "XGBoost": xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, gamma=0,
        reg_alpha=0.1, reg_lambda=1, random_state=0
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1, random_state=0
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_split=5, min_samples_leaf=2, subsample=0.8,
        random_state=0
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=0
    ),
    "CatBoost": CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        l2_leaf_reg=1, verbose=0,subsample=0.8,random_state=0
    ),
    "DecisionTree": DecisionTreeRegressor(
        max_depth=10, min_samples_split=5, min_samples_leaf=2,
        random_state=0
    )
}
```
2. Group 2: more estimators
```py
models = {
    "XGBoost": xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, gamma=0,
        reg_alpha=0.1, reg_lambda=1, random_state=0
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1, random_state=0
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=5,
        min_samples_split=5, min_samples_leaf=2, subsample=0.8,
        random_state=0
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=1000, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=0
    ),
    "CatBoost": CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=6,
        l2_leaf_reg=1, subsample=0.8, verbose=0,random_state=0
    ),
    "DecisionTree": DecisionTreeRegressor(
        max_depth=10, min_samples_split=5, min_samples_leaf=2,
        random_state=0
    )
}
```
3. Group 3: No regularization
```py
models = {
    "XGBoost": xgb.XGBRegressor(
         random_state=0
    ),
    "LightGBM": lgb.LGBMRegressor(
        random_state=0
    ),
    "GradientBoosting": GradientBoostingRegressor(
       
        random_state=0
    ),
    "RandomForest": RandomForestRegressor(
         random_state=0
    ),
    "CatBoost": CatBoostRegressor(
        random_state=0
    ),
    "DecisionTree": DecisionTreeRegressor(
        random_state=0
    )
}
```

### Man results
-----------------
*Table 11. Comparision of Group 1 model with man dataset. GradientBoost and CatBoost shows relatively high precision and recall at 10 on personal characteristic*

| Model         | Train RMSE | Test RMSE | Test Acc | Test Precision | Test Recall | Precision@10 | Recall@10 | Personal Precision@10 | Personal Recall@10 |
| ------------- | ---------- | --------- | -------- | -------------- | ----------- | ------------- | ---------- | -------------- | -------------- |
| XGBoost       | 0.5816     | 0.7459    | 0.63     | 0.85           | 0.29        | 0.1340        | 0.7724     | 0.6424         | 0.4697         |
| LightGBM      | 0.6572     | 0.7423    | 0.64     | 0.87           | 0.28        | 0.1340        | 0.7724     | 0.6424         | 0.4694         |
| GradientBoost | 0.6748     | 0.7373    | 0.64     | 0.87           | 0.28        | 0.1340        | 0.7724     | 0.6485         | 0.4727         |
| RandomForest  | 0.6799     | 0.7481    | 0.62     | 0.87           | 0.24        | 0.1340        | 0.7724     | 0.6242         | 0.4651         |
| CatBoost      | 0.6727     | 0.7314    | 0.63     | 0.89           | 0.27        | 0.1340        | 0.7724     | 0.6576         | 0.4800         |
| DecisionTree  | 0.7086     | 0.8034    | 0.63     | 0.81           | 0.30        | 0.1340        | 0.7724     | 0.6061         | 0.4485         |

-----------------
*Table 12. Comparision of Group 2 model with man dataset. XGBoost, GradientBoost and CatBoost shows relatively high precision and recall at 10 on personal characteristic*

| Model         | Train RMSE | Test RMSE | Test Acc | Test Precision | Test Recall | Precision@10 | Recall@10 | Personal Precision@10 | Personal Recall@10 |
| ------------- | ---------- | --------- | -------- | -------------- | ----------- | ------------- | ---------- | -------------- | -------------- |
| XGBoost       | 0.5003     | 0.7623    | 0.64     | 0.83           | 0.30        | 0.1340        | 0.7724     | 0.6455         | 0.4701         |
| LightGBM      | 0.6070     | 0.7510    | 0.64     | 0.85           | 0.29        | 0.1340        | 0.7724     | 0.6455         | 0.4700         |
| GradientBoost | 0.5869     | 0.7502    | 0.64     | 0.83           | 0.31        | 0.1340        | 0.7724     | 0.6424         | 0.4697         |
| RandomForest  | 0.6796     | 0.7483    | 0.62     | 0.87           | 0.24        | 0.1340        | 0.7724     | 0.6242         | 0.4632         |
| CatBoost      | 0.6179     | 0.7348    | 0.63     | 0.85           | 0.28        | 0.1340        | 0.7724     | 0.6485         | 0.4734         |
| DecisionTree  | 0.7086     | 0.8034    | 0.63     | 0.81           | 0.30        | 0.1340        | 0.7724     | 0.6061         | 0.4485         |

-----------------

*Table 13. Comparision of Group 3 model with man dataset. XGBoost, GradientBoost and CatBoost shows relatively high precision and recall at 10 on personal characteristic, while RandomForest and Decision Tree suffer from overfitting*

| Model            | Train RMSE | Test RMSE | Test Acc | Test Precision | Test Recall | Precision@10 | Recall@10 | Personal Precision@10 | Personal Recall@10 |
| ---------------- | ---------- | --------- | -------- | -------------- | ----------- | ------------- | ---------- | -------------- | -------------- |
| XGBoost      | 0.5647     | 0.7671    | 0.64     | 0.85           | 0.31        | 0.1340        | 0.7724     | 0.6394         | 0.4654         |
| LightGBM     | 0.6930     | 0.7389    | 0.64     | 0.90           | 0.27        | 0.1340        | 0.7724     | 0.6424         | 0.4657         |
| GradientBoost    | 0.7469     | 0.7416    | 0.62     | 0.89           | 0.24        | 0.1340        | 0.7724     | 0.6424         | 0.4715         |
| RandomForest   | 0.3043     | 0.7593    | 0.63     | 0.86           | 0.28        | 0.1340        | 0.7724     | 0.6212         | 0.4631         |
| CatBoost     | 0.6095     | 0.7348    | 0.64     | 0.87           | 0.29        | 0.1340        | 0.7724     | 0.6485         | 0.4724         |
| DecisionTree | 0.0933     | 1.0773    | 0.68     | 0.67           | 0.66        | 0.1340        | 0.7724     | 0.5606         | 0.4356         |

### Woman results

*Table 14. Comparision of Group 1 model with woman dataset. CatBoost shows relatively high precision and recall at 10 on personal characteristic*

| Model | Train RMSE | Test RMSE | Test Accuracy | Test Precision | Test Recall | Precision@10 | Recall@10 | Personal Precision@10 | Personal Recall@10 |
|-------|------------|-----------|---------------|----------------|-------------|--------------|-----------|---------------------|-------------------|
| XGBoost | 0.635 | 0.773 | 0.56 | 0.84 | 0.29 | 0.139 | 0.839 | 0.607 | 0.629 |
| LightGBM | 0.703 | 0.771 | 0.55 | 0.84 | 0.27 | 0.139 | 0.839 | 0.602 | 0.625 |
| GradientBoosting | 0.719 | 0.769 | 0.55 | 0.86 | 0.26 | 0.139 | 0.839 | 0.605 | 0.622 |
| RandomForest | 0.722 | 0.773 | 0.53 | 0.87 | 0.22 | 0.139 | 0.839 | 0.609 | 0.624 |
| CatBoost | 0.719 | 0.767 | 0.55 | 0.86 | 0.26 | 0.139 | 0.839 | 0.614 | 0.630 |

--------------------
*Table 15. Comparision of Group 2 model with woman dataset. XGBoost shows relatively high precision and recall at 10 on personal characteristic*

| Model | Train RMSE | Test RMSE | Test Accuracy | Test Precision | Test Recall | Precision@10 | Recall@10 | Personal Precision@10 | Personal Recall@10 |
|-------|------------|-----------|---------------|----------------|-------------|--------------|-----------|---------------------|-------------------|
| XGBoost | 0.559 | 0.785 | 0.57 | 0.83 | 0.31 | 0.139 | 0.839 | 0.614 | 0.631 |
| LightGBM | 0.659 | 0.775 | 0.56 | 0.84 | 0.29 | 0.139 | 0.839 | 0.605 | 0.625 |
| GradientBoosting | 0.641 | 0.779 | 0.56 | 0.85 | 0.30 | 0.139 | 0.839 | 0.612 | 0.630 |
| RandomForest | 0.721 | 0.773 | 0.53 | 0.87 | 0.22 | 0.139 | 0.839 | 0.614 | 0.627 |
| CatBoost | 0.672 | 0.771 | 0.56 | 0.85 | 0.28 | 0.139 | 0.839 | 0.612 | 0.632 |
| DecisionTree | 0.752 | 0.823 | 0.53 | 0.81 | 0.24 | 0.139 | 0.839 | 0.563 | 0.601 |

---------------------------------
*Table 15. Comparision of Group 3 model with woman dataset. Catboost shows relatively high precision and recall at 10 on personal characteristic, while RandomForest and DecisionTree suffer from overfitting*

| Model | Train RMSE | Test RMSE | Test Accuracy | Test Precision | Test Recall | Precision@10 | Recall@10 | Personal Precision@10 | Personal Recall@10 |
|-------|------------|-----------|---------------|----------------|-------------|--------------|-----------|---------------------|-------------------|
| XGBoost | 0.621 | 0.796 | 0.57 | 0.84 | 0.32 | 0.139 | 0.839 | 0.600 | 0.623 |
| LightGBM | 0.734 | 0.769 | 0.55 | 0.87 | 0.26 | 0.139 | 0.839 | 0.607 | 0.624 |
| GradientBoosting | 0.778 | 0.770 | 0.53 | 0.88 | 0.21 | 0.139 | 0.839 | 0.600 | 0.621 |
| RandomForest | 0.314 | 0.773 | 0.57 | 0.86 | 0.30 | 0.139 | 0.839 | 0.607 | 0.622 |
| CatBoost | 0.659 | 0.774 | 0.56 | 0.84 | 0.29 | 0.139 | 0.839 | 0.612 | 0.629 |
| DecisionTree | 0.087 | 1.082 | 0.64 | 0.70 | 0.66 | 0.139 | 0.839 | 0.544 | 0.596 |

## Lightweight Face shape classification 
We reduce the face shape classification model size using EfficientNetB2. We change the input image size to 260x260, and add dropout(0.3) for preventing overfiting We initially trained the model with the EfficientNetB2 backbone frozen for 25 epochs, learning rate 1e-3(with cosine decay) and AdamW(weight_decay=5e-5), and then fine-tuned the entire network by unfreezing all layers and using a low learning rate(3e-5 with cosine decay) and AdamW(weight_decay=5e-6) for 30 epochs. Earlystopping parameters are same as previous model.

*Table 16. Comparision of original and lightweight model. results show that EfficientNetB2 model achieve similar accuracy while has small size*

| model version                 | test loss | test accuracy | Model size |
|-------------------------------|-----------|---------------|------------|
| Ver2(EfficientNetB4 backbone) |   1.2050  |     75.83%    |   17.68M   |
| Ver3(EfficientNetB2 backbone) |   1.5023  |     75.13%    |    7.78M   |

## Increasing accuracy by data augmentation and label smoothing
Inspired by (2), We totally renew the face shape classification code(ver4, ver5, ver6) with data augmentation, label smoothing, learning rate 5e-3 and batch size 32 for 30 epochs. The results show that data augmentation and label smoothing helps increasing accuracy.

*Table 16. Comparision of models. The results show that data augmentation and label smoothing helps increasing accuracy*

| model version                 | test loss | test accuracy | Model size |
|-------------------------------|-----------|---------------|------------|
| Ver2(EfficientNetB4 backbone) |   1.2050  |     75.83%    |   17.68M   |
| Ver3(EfficientNetB2 backbone) |   1.5023  |     75.13%    |    7.78M   |
| Ver4(EfficientNetB4 backbone with (2)) |   0.7366  |     83.40%    |    17.68M  |
| Ver5(EfficientNetB2 backbone with (2)) |   0.7444  |     83.60%    |    7.78M   |
| Ver6(EfficientNetB0 backbone with (2)) |   0.8224  |     81.10%    |    4.06M   |


(2) [https://www.kaggle.com/code/baranbingl/face-shape-detection-85-acc-on-test-set](https://www.kaggle.com/code/baranbingl/face-shape-detection-85-acc-on-test-set)

You can view the Ver5 code on the [Kaggle](https://www.kaggle.com/code/minchocoin/tensorflow-efficientnetb2-test-accuracy-83percent). Due to the random state, results on Colab(Table 16) and Kaggle are slightly different(test accuracy of Kaggle are 83.30%)

## Conclusion
We calculate precision, recall, precision at k, recall at k. Also, we draw the precision-recall curve. above results show that the recommendation system can recommend the clothes based on the personal characteristic (personal color, faceshape, bodyshape). Additionally, we compare the recommendation systsm based on various machine learning model. results show that XGBoost, GradientBoosting, and Catboost shows relatively high precision at 10 and recall at 10 on personal characteristics.
Also, We lightweight the face shape classification model using EfficientNetB2. results show that lightweight model achieve similar accuracy while reducing the model size. Additionally, we renew the face shape classification model(ver4, ver5, ver6) model. results show that data augmentation helps increasing accuracy.

# BibTex
```
@software{Fashion_recommendation_personal,
  author = {Seonghoon Lee and Taehun Kim},
  month = {11},
  title = {Fashion recommendation system based on personal characteristics},
  url = {https://github.com/pnucse-capstone-2024/Capstone-2024-team-24},
  year = {2024}
}
```

```
@misc{Fashion_recommendation_personal_webpage,
  title = {Fashion recommendation system based on personal characteristics webpage},
  author = {Seonghoon Lee and Taehun Kim},
  year = {2025},
  url = {https://minchocoin.github.io/capstone-team-24-page/},
  note = {Accessed on May 27, 2025}
}
```
# Authors
- [Seonghoon Lee](https://github.com/NextrPlue)
    - B.S in electrical and computer engineering from Pusan National University
- [Taehun Kim](https://github.com/minchoCoin)
    - B.S in electrical and computer engineering from Pusan National University