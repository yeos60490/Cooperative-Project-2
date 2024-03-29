#  Cooperative Project II
여행지 추천 웹 서비스.
사용자가 웹에서 자신의 성향 혹은 사진을 입력하면 머신러닝을 통해 여행지를 추천해주는 서비스입니다. 

- 서비스의 차별성 
  1. 대부분의 여행 서비스들은 검색 기반의 정적인 서비스를 제공.
  2. 시대의 흐름에 맞춰 지능적인 AWS 기술을 도입하여 개인에게 최적화된 서비스를 제공

- 서비스 사례 예시
  1. 따뜻한 나라로 가서 활동적인 여행을 즐기고 싶지만 그 나라가 우기인지 아닌지 혹은 즐길 만한 요소들이 충분한지에 대해서 직접 알아봐야하는 번거로움이 있습니다. 그러나 VoyageGayage 서비스를 이용한다면 개인의 여행 스타일을 고려한 여행지 추천과 함께 여행지에 대한 정보도 한번에 확인 할 수 있습니다.

  2. 여행이 대중화 됨에 따라 유명한 관광지는 관광객들로 붐벼 기대한만큼 즐기거나 쉬고 오지 못하는 경우가 있습니다. 이에 많은 사람들이 알지 못하는 여행지를 찾아가려 하고 있습니다. 이러한 사람들의 성향에 맞춰 가려던 유명한 여행지와 비슷한 대안의 여행지를 추천 받으실 수 있습니다.
  
-------------------------
- 사용 기술


      AWS Lambda
      AWS Sagemaker
      AWS Rekognition
      AWS Cognito, Api-gateway
      AWS S3, DynamoDB
      Beautifulsoup
      Google geocoding, place API
      Firebase Real-time DB
    
    
- 담당


      프레임워크 설계
      설문조사를 통한 데이터 수집
      DB, Storage에 데이터 저장
      AWS Sagemaker 사용하여 ML 구현
      서비스 상호작용을 위한 AWS Lambda 구현
      ML 점진적 학습 구현
      Google API 사용하여 추천 UI 구현
    
    
- Framework

<img width="80%" src="https://user-images.githubusercontent.com/54983764/132698531-7262ec00-05aa-4234-a633-d726aedda713.png"/>
<img width="80%" src="https://user-images.githubusercontent.com/54983764/132698864-5de34074-c119-4aa6-89f6-8f810ad67c6d.png"/>

- Web

### 설문조사를 통한 여행지 추천
<img width="80%" src="https://user-images.githubusercontent.com/54983764/132822339-99fd8f49-0972-4773-b1fa-bbb966373b86.gif"/>

### 이미지를 통해 비슷한 장소 추천
<img width="80%" src="https://user-images.githubusercontent.com/54983764/132822857-ec4a3ae7-d84b-4e22-86a3-16091359e846.gif"/>
