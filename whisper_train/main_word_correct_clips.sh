# deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag xunfei_202409_1 --last_saved clean_57 > log/main_word_xf1.log 2>&1
# deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag xunfei_202409_2 --last_saved xunfei_202409_1 > log/main_word_xf2.log 2>&1
# deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag WORD_CORRECT_1 --last_saved xunfei_202409_2 > log/main_word_wc1.log 2>&1
# deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag WORD_CORRECT_2 --last_saved WORD_CORRECT_1 > log/main_word_wc2.log 2>&1
# deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag WORD_CORRECT_3 --last_saved WORD_CORRECT_2 > log/main_word_wc3.log 2>&1
# deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag WORD_CORRECT_4 --last_saved WORD_CORRECT_3 > log/main_word_wc4.log 2>&1
# deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag WORD_CORRECT_5 --last_saved WORD_CORRECT_4 > log/main_word_wc5.log 2>&1
# deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag WORD_CORRECT_6 --last_saved WORD_CORRECT_5 > log/main_word_wc6.log 2>&1
# deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag WORD_CORRECT_7 --last_saved WORD_CORRECT_6 > log/main_word_wc7.log 2>&1
# deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag WORD_CORRECT_8 --last_saved WORD_CORRECT_7 > log/main_word_wc8.log 2>&1
 
# deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag WORD_CORRECT_9 --last_saved WORD_CORRECT_8 > log/main_word_wc9.log 2>&1
# deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag WORD_CORRECT_10 --last_saved WORD_CORRECT_9 > log/main_word_wc10.log 2>&1
# deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag WORD_CORRECT_11 --last_saved WORD_CORRECT_10 > log/main_word_wc11.log 2>&1

deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag WORD_CORRECT_12 --last_saved WORD_CORRECT_11 > log/main_word_wc12.log 2>&1
deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag WORD_CORRECT_13 --last_saved WORD_CORRECT_12 > log/main_word_wc13.log 2>&1
deepspeed --include localhost:2,3,4,5 main_word_correct_clips.py --batch_tag WORD_CORRECT_14 --last_saved WORD_CORRECT_13 > log/main_word_wc14.log 2>&1


