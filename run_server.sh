

python whisper_fastapi_online_server.py \
--model large-v3-turbo \
--lan fr \
--task transcribe \
--vac \
--backend mlx-whisper \
--buffer_trimming sentence \
--warmup-file samples_jfk.wav \
$@