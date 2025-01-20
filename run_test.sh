#python whisper_online.py \
python translation_online.py \
--model large-v3-turbo \
--lan fr \
--task transcribe \
--backend mlx-whisper \
--buffer_trimming sentence \
--min-chunk-size 3\
 --buffer_trimming_sec 5 \
--audio_path preche.wav \
--vac \
--vac-chunk-size 2 \
$argv

