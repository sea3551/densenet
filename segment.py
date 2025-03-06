import os
import pandas as pd
import numpy as np

# 1️⃣ 입력 및 출력 폴더 설정
input_dir = 'single_data/day1/mi'              # 원본 CSV 파일들이 있는 폴더
output_dir = '400_0.5_segment_data/day1/mi'            # 슬라이딩 윈도우 처리 후 저장할 폴더

# 출력 폴더가 없다면 생성
os.makedirs(output_dir, exist_ok=True)

# 2️⃣ 슬라이딩 윈도우 설정
window_size = 400    # 200ms (샘플링 주파수에 따라 조정 필요)
overlap = 0.5        # 50% 오버랩
step_size = int(window_size * (1 - overlap))  # 실제 이동 거리 (100ms)

# 3️⃣ 슬라이딩 윈도우 함수 정의
def sliding_window(data, window_size, step_size):
    segments = []
    start = 0
    
    while (start + window_size) <= len(data):
        segment = data.iloc[start:start + window_size]
        segments.append(segment)
        start += step_size  # 스텝 크기만큼 이동

    return segments


# 4️⃣ 디렉토리 내 모든 CSV 파일 처리
file_counter = 1  # 파일 이름 인덱스 초기화

for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        print(f"📥 처리 중: {filename}")

        # CSV 파일 불러오기
        data = pd.read_csv(file_path)

        # 슬라이딩 윈도우 적용 (전체 데이터)
        segments = sliding_window(data, window_size, step_size)

        # 5️⃣ 각 세그멘트를 개별 CSV 파일로 저장
        for segment in segments:
            output_filename = f"segment_mi_{file_counter}.csv"  # 파일 이름 형식 변경
            output_path = os.path.join(output_dir, output_filename)

            segment.to_csv(output_path, index=False)
            print(f"✅ 저장 완료: {output_path}")

            file_counter += 1  # 파일 인덱스 증가

print("🎯 모든 CSV 파일 슬라이딩 윈도우 처리 및 저장 완료!")
