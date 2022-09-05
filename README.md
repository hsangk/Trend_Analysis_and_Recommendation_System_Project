# 이미지 및 텍스트 마이닝을 통한 사용자 분석과 장소 추천 서비스 (Trend Analysis and Recommendation System based on Image and TextMining)
## Project
2022-1 데이터사이언스 캡스톤디자인

## Packages
1. Selenium
2. BeautifulSoup
3. konlpy
4. pandas
5. numpy
6. captum
7. matplotlib
8. sklearn 

## WebCrawling Dev.

### 실행
1. 네이버 플레이스 기본 정보 수집 
```
naver_test.py
```

2. 카카오 맵 기본 정보 수집
```
kakao_test.py
```

## WebPage Dev.

### 실행
1. 경로 설정
```
cd Webpage
```

2. 가상환경 접속
```
pipenv shell
```

3. 라이브러리 설치
```
pip install -r requirements.txt
```

4. 프로젝트 경로 이동
```
cd myproject
```

5. database 설정

- 기본 세팅 : 

https://heroeswillnotdie.tistory.com/16

- myproject/main/database.py 생성 
- 데이터베이스 연동
```
python main/database.py
```

6. migrate
```
python manage.py makemigrations
python manage.py migrate
```

7. runserver
```
python manage.py runserver
```
## Image clustering.
1. 장소 이미지 별 rgb값 클러스터링 -> 대표값 추출 
```
3.restaurant_clutering.py
```
```
### 3.restaurant_clustering.py 

def rgb_perc(k_cluster):
    # width = 300
    # palette = np.zeros((50, width, 3), np.uint8)

    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_)  # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i] / n_pixels, 2)
    perc = np.array(list(perc.values()))

    # top5 color * perc = average RGB
    rgb_weight = k_cluster.cluster_centers_.T * perc
    rgb_avg = np.mean(rgb_weight, axis=1)

    # print('Percentage of color :', perc)
    # print('Each RGB :', k_cluster.cluster_centers_)
    # print('Avg_RGB :',rgb_avg)
    step = 0

    # for idx, centers in enumerate(k_cluster.cluster_centers_):
    #     palette[:, step:int(step + perc[idx]*width+1), :] = centers
    #     step += int(perc[idx]*width+1)

    return rgb_avg
```
2. 대표 rgb값 기반 장소 클러스터링 
```
4.cluter_result.py
```
```
### 4.cluter_result.py

from sklearn.cluster import KMeans

clust_model = KMeans(n_clusters = 5 # 클러스터 갯수
                     # , n_init=10 # initial centroid를 몇번 샘플링한건지, 데이터가 많으면 많이 돌릴수록 안정화된 결과가 나옴
                     # , max_iter=500 # KMeans를 몇번 반복 수행할건지, K가 큰 경우 1000정도로 높여준다
                     # , random_state = 42 # , algorithm='auto'
                     )

# 생성한 모델로 데이터를 학습
clust_model.fit(df_f) # unsupervised learning

# 결과 값을 변수에 저장
centers = clust_model.cluster_centers_ # 각 군집의 중심점
pred = clust_model.predict(df_f) # 각 예측군집
print(pd.DataFrame(centers))
print(pred[:10])

clust_df2 = df_f.copy()
clust_df2['clust'] = pred
clust_df2.to_csv('\Inner_image_clustering\\result.csv',index= False)
```

### 결과

- AWS EC2 배포 결과 : http://mudsil.com/

<img width="1375" alt="image" src="https://user-images.githubusercontent.com/87521259/180601272-94f6b0eb-198a-45aa-943c-1d17c6562f4a.png">
<img width="1372" alt="스크린샷 2022-07-23 오후 7 24 57" src="https://user-images.githubusercontent.com/87521259/180601288-82a6820b-76e5-4419-87e6-ac6e194cde07.png">
<img width="1368" alt="스크린샷 2022-07-23 오후 7 25 20" src="https://user-images.githubusercontent.com/87521259/180601296-1dfb975d-9e8f-411b-b5ab-142019d37bd4.png">

- 2022 데이터사이언스 캡스톤디자인 1등
