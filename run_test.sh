#python whisper_online.py \
python translation_online.py \
--model large-v3-turbo \
--lan fr \
--task transcribe \
--vac \
--backend mlx-whisper \
--buffer_trimming sentence \
--audio_path preche.wav \
$argv
