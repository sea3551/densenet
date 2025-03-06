import os
import pandas as pd

# 1️⃣ 입력 및 출력 폴더 설정
input_dir = '400_0.5_normalize_data/day1/mi'        # 원본 CSV 파일들이 있는 폴더
output_dir = '400_0.5_merge_normal_data/day1/mi'           # 병합된 데이터 저장 폴더

# 출력 폴더가 없다면 생성
os.makedirs(output_dir, exist_ok=True)

# 2️⃣ 디렉토리 내 모든 CSV 파일 처리
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        print(f"📥 데이터 처리 중: {filename}")

        # CSV 파일 불러오기
        data = pd.read_csv(file_path)

        # 3️⃣ Comp Ch 3와 Comp Ch 4를 수직으로 병합
        if 'Comp Ch 3' in data.columns and 'Comp Ch 4' in data.columns:
            merged_data = pd.concat([data['Comp Ch 3'], data['Comp Ch 4']], ignore_index=True)
            
            # 4️⃣ 새로운 데이터프레임으로 변환
            merged_df = pd.DataFrame({'Merge Ch': merged_data})

            # 5️⃣ 병합된 데이터 저장
            base_filename = os.path.splitext(filename)[0]
            output_filename = f"merge_{base_filename}.csv"
            output_path = os.path.join(output_dir, output_filename)

            merged_df.to_csv(output_path, index=False)
            print(f"✅ 저장 완료: {output_path}")
        else:
            print(f"⚠️ {filename} 파일에 'Comp Ch 3' 또는 'Comp Ch 4' 컬럼이 없습니다.")

print("🎯 모든 CSV 파일의 열 병합 및 저장 완료!")
