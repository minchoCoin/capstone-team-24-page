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
This model uses cosine similarity to find similar clothing items based on keyword of the clothes. We use TF-IDF Vectorizing to embedding the keyword. We collected (clothes - keyword) data from [K-Fashion](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=51) data. We assign fashion keywords based on specific physical characteristics and recommend clothing items that share similar keywords. Alternatively, we recommend items that match the keywords entered by the user.


# Applications
This technology enables highly personalized fashion recommendations that consider individual physical attributes rather than just following general trends. The web interface allows users to upload photos or manually input their characteristics to receive tailored fashion suggestions.
# Future Directions
Future enhancements include expanding the dataset, incorporating real-time trend analysis, and potential integration with e-commerce platforms to create a seamless shopping experience.

---

# Data
- [fashion preferences and recommendation data by year](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71446)

- [K-Fashion](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=51)

- [Face Shape dataset](https://www.kaggle.com/datasets/niten19/face-shape-dataset)
- [Human Body classification dataset](https://data.ncl.ac.uk/articles/dataset/Human_Body_Shape_Classification_Dataset/19307300?file=34292915)

- [Korean bodyshape data](https://sizekorea.kr/human-info/body-shape-class/age-gender-body)

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

# References
1. Yun-Seok Jung, “A Study on the Quantitative Diagnosis Model of Personal 
Color,” Journal of Convergence for Information Technology, Vol. 11, No. 11, pp. 
277-287, 2021. (in Korean)
2. Jong-Suk An, “A Study on Effective Image Making Depending on Hair Style 
and Neckline,” J Korean Soc Cosmetol, Vol. 15, No. 1, pp.342-351, 2009. (in 
Korean)
3. Soo-ae Kwon, Fashion and Life, Gyohakyungusa, 2016.
4. “2023 Consumption Trend Series - 03 Personalized Services,” MezzoMedia 
Available: [https://www.mezzomedia.co.kr/data/insight_m_file/insight_m_file_1605.pdf](https://www.mezzomedia.co.kr/data/insight_m_file/insight_m_file_1605.pdf)
 (downloaded 2024, May. 19)
5. So-young Lee, “Personal Color Tone Type and Categorization of Harmonious 
Colors According to Skin Color,” M.S. thesis, Graduate School of Cultural and 
Information Policy, Hongik Univ., Seoul, South Korea, 2019. (in Korean)
6. Cameron Patrick Trotter et al. Human Body Shape Classification Dataset. 
[https://data.ncl.ac.uk/articles/dataset/Human_Body_Shape_Classification_Dataset/19307300?file=34292915](https://data.ncl.ac.uk/articles/dataset/Human_Body_Shape_Classification_Dataset/19307300?file=34292915)