import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# ✅ 1️⃣ CSV → 스펙트로그램 변환 함수
def csv_to_spectrogram(file_path, output_dir, fs=1000, fmin=20, hop_length=4):
    # 데이터 불러오기
    emg_signal = np.loadtxt(file_path, delimiter=',', skiprows=1)
        
    # CQT 하이퍼파라미터 설정
    bins_per_octave = 12
    num_octaves = 4
    n_bins = bins_per_octave * num_octaves

    # ✅ CQT 변환
    cqt_result = np.abs(librosa.cqt(emg_signal, sr=fs, fmin=fmin,
                                    n_bins=n_bins, bins_per_octave=bins_per_octave,
                                    hop_length=hop_length))
    cqt_db = librosa.amplitude_to_db(cqt_result, ref=np.max)
    
    # 파일 이름 추출 및 저장 경로 설정
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{file_name}.png")
    
    # ✅ 스펙트로그램 시각화 및 저장
    plt.figure(figsize=(4, 4))  # CNN 학습용으로 정사각형 이미지
    librosa.display.specshow(cqt_db, sr=fs, x_axis='time', y_axis='linear', cmap='magma')
    plt.ylim([20, fs / 2])  # ✅ 최소 20Hz, 최대 500Hz로 제한
    plt.axis('off')  # CNN을 위해 축 제거
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# ✅ 2️⃣ 디렉토리 내 모든 CSV 처리
def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_dir, file_name)
            print(f"📊 Processing {file_name}...")
            csv_to_spectrogram(file_path, output_dir)

# ✅ 3️⃣ 경로 설정 및 실행
<<<<<<< HEAD
input_directory = '400_0.5_augment_data/day1/dae'  # CSV 파일이 있는 폴더
output_directory = 'spectrogram_images/day1/dae'          # 스펙트로그램 이미지 저장 폴더
=======
input_directory = '/Users/syj/Desktop/code/augment_data/day1/dae'  # CSV 파일이 있는 폴더
output_directory = '/Users/syj/Desktop/code/spectrogram_images/day1/dae'          # 스펙트로그램 이미지 저장 폴더
>>>>>>> 9c3ce225915b41c7829d936d178d26046cbb0087

process_directory(input_directory, output_directory)
print("✅ 모든 CSV 파일이 스펙트로그램 이미지로 변환되었습니다!")
