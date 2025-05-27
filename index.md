---
layout: project_page
permalink: /

title: Fashion recommendation system based on personal characteristics
authors:
    이성훈, 김태훈
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
We use an EfficientNetB4-based deep learning model. The model was built based on the EfficientNetB4 network pretrained on the ImageNet dataset. A GlobalAveragePooling2D layer was applied to the output of EfficientNetB4, and the resulting output was passed through a Fully-connected layer to classify into five face shapes. We initially trained the model with the EfficientNetB4 backbone frozen, and then fine-tuned the entire network by unfreezing all layers and using a low learning rate. The system identifies five face shapes (heart, long, oval, round, square) with 75.8% accuracy.  These classifications inform recommendations for clothing styles and necklines that complement facial features.
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

# Appendix
After the capstone project is finished, we evaluate the recommendation system more precisely

We considered a score of 3 or higher to be preferred (or recommended) and a score of less than 3 to be disliked (or not recommended) based on the score the user gave to the clothes and the score predicted by the recommendation system.

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

$$ Precision @k = \frac{\text{Number of items relevant to the user in top K}}{K}$$

$$ Recall @ k = \frac{\text{Number of items relevant to the user in top K}}{\text{Total number of items relevant to the user}}$$
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

## Limitation measuring precision with test dataset

As shown in the Figure 32 and 34, in the test dataset, many of the users  rated only 1-2 clothes. This makes the precision and recall inaccurate. Therefore, the results of the test data above must be interpreted carefully.

![fig110](/static/image/figure110.png)

*Figure 31. Distribution of number of ratings per user in man training dataset*

![fig111](/static/image/figure111.png)

*Figure 32. Distribution of number of ratings per user in man test dataset*

![fig112](/static/image/figure112.png)

*Figure 33. Distribution of number of ratings per user in woman training dataset*

![fig113](/static/image/figure113.png)

*Figure 34. Distribution of number of ratings per user in woman test dataset*

## Conclusion
We calculate precision, recall, precision at k, recall at k. Also, we draw the precision-recall curve. above results show that the recommendation system can recommend the clothes based on the personal characteristic (personal color, faceshape, bodyshape)

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