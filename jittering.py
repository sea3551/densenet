import os
import pandas as pd
import numpy as np

# 1️⃣ 입력 및 출력 폴더 설정
input_dir = '400_0.5_merge_normal_data/day1/mi'        # 원본 CSV 파일들이 있는 폴더
output_dir = '400_0.5_augment_data/day1/mi'     # 지터링 적용 후 저장할 폴더

# 출력 폴더가 없다면 생성
os.makedirs(output_dir, exist_ok=True)

# 2️⃣ 지터링 함수 정의 (노이즈 추가)
def jittering(data, noise_level=0.05):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

# 3️⃣ 디렉토리 내 모든 CSV 파일 처리
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        print(f"📥 지터링 적용 중: {filename}")

        # CSV 파일 불러오기
        data = pd.read_csv(file_path)

        # Merge Ch에 지터링 적용
        jittered_data = data.copy()
        if 'Merge Ch' in jittered_data.columns:
            jittered_data['Merge Ch'] = jittering(
                jittered_data['Merge Ch'].values, noise_level=0.05
            )

        # 4️⃣ 기존 파일 이름을 포함하여 저장
        base_filename = os.path.splitext(filename)[0]  # 확장자 제거 (예: 'mi_1')
        output_filename = f"aug_{base_filename}.csv"   # 새로운 파일 이름 생성
        output_path = os.path.join(output_dir, output_filename)

        jittered_data.to_csv(output_path, index=False)
        print(f"✅ 저장 완료: {output_path}")

print("🎯 모든 CSV 파일 지터링 데이터 증강 완료!")

