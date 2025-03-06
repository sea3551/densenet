import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1️⃣ 입력 및 출력 폴더 설정
input_dir = '400_0.5_segment_data/day1/mi'        # 원본 CSV 파일들이 있는 폴더
output_dir = '400_0.5_normalize_data/day1/mi'    # 정규화된 파일 저장 폴더

# 출력 폴더가 없다면 생성
os.makedirs(output_dir, exist_ok=True)

# 2️⃣ Min-Max 스케일러 정의
scaler = MinMaxScaler()

# 3️⃣ 디렉토리 내 모든 CSV 파일 처리
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        print(f"📥 Min-Max 정규화 적용 중: {filename}")

        # CSV 파일 불러오기
        data = pd.read_csv(file_path)

        # Comp Ch 3와 Comp Ch 4에 Min-Max 정규화 적용
        normalized_data = data.copy()
        if 'Comp Ch 3' in data.columns and 'Comp Ch 4' in data.columns:
            normalized_data[['Comp Ch 3', 'Comp Ch 4']] = scaler.fit_transform(
                data[['Comp Ch 3', 'Comp Ch 4']]
            )

            # 4️⃣ 정규화된 데이터 저장
            base_filename = os.path.splitext(filename)[0]  # 확장자 제거
            output_filename = f"normalize_{base_filename}.csv"  # 새로운 파일 이름
            output_path = os.path.join(output_dir, output_filename)

            normalized_data.to_csv(output_path, index=False)
            print(f"✅ 저장 완료: {output_path}")
        else:
            print(f"⚠️ 파일 {filename}에 'Comp Ch 3' 또는 'Comp Ch 4' 컬럼이 없습니다.")

print("🎯 모든 CSV 파일 Min-Max 정규화 완료!")
