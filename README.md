# Image-compression-program-that-preserves-channel-information
본 repository는 한국 저작권 위원회에 등록 된 '채널 정보를 보존한 이미지 압축 프로그램'의 python 코드를 저장하기 위한 목적으로 개설 되었습니다.

각 폴더와 파일에 대한 설명은 다음과 같습니다.
data: 유저가 프로그램을 실행하기 위해 이미지 데이터를저장하는폴더입니다.
이미지 데이터는 (데이터수,높이,넓이,채널수) 혹은 (데이터수,넓이,높이,채널수)의 npy(numpy)형식의 이미지 데이터를 저장해야합니다.

processed_data:압축된 데이터가 저장되는 폴더입니다.

data_utils.py:데이터로드및훈련(train),검증(validation),시험(test) 데이터 분리 기능을 제공하는 py파일입니다.

make_pca_data.py:채널 정보를 보존한 pca를 진행하는 py파일입니다.압축된 데이터는 processed_data폴더에 저장됩니다.

three_machine_learning.py:압축된 데이터를 SVM,LinearRegression,MLP 이용하여 학습 및 성능평가를 진행하는 py파일입니다.

argument.py:학습에 필요한 인자 및 다양한 커스터마이징을 위한 인자를 저장하는 py파일입니다.
유저는 해당 파일의 인자들을 제공하는 설명대로 수정함으로써 다양한 환경에서 실험을 진행할수있습니다.

run.py:최종적인 실행을 위한 파일입니다.
폴더 내 파일들은 유기적으로 실행되므로 이미지 데이터 저장 후 해당 파일만 실행하면 데이터 압축 및 기계학습3종에서의 성능 평가를 그래프와 출력창을 통해 제공받을 수 있습니다.

# Information

Affiliation: LearnDataLab, SKKU
E-mail : rnrnfjwl11@skku.edu
