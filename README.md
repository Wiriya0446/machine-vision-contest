predict_per_menu_no_answer.py         => ตรวจคำตอบโดยที่ไฟล์ csv ไม่มีเฉลย (ได้ output เป็นไฟล์ csv และแสดงผลผ่าน terminal)

predict_per_menu.py                   => ตรวจคำตอบโดยที่ไฟล์ csv มีเฉลย (ได้ output เป็นไฟล์ csv และแสดงผลผ่าน terminal)

train_per_menu.py                     => ที่ใช้เทรนโมเดล

auto_label.py                         => auto label ภาพจาก Intragram Images โดยการสุ่มจับคู่ภาพที่เป็นเมนูเดียวกันมาอยู่ด้วยกัน

filter_and_balance.py                 => กรองภาพจาก auto label คัดเอาเฉพาะคู่รูปภาพที่มีความมั่นใจสูง

data_from_intragram_2.csv             => dataset จาก IG ที่ถูกสุ่มสลับตำแหน่งภาพที่น่ากินและแบ่งออกไป 100 คู่เพื่อใช้ในการตรวจ

Test_IG.csv                           => dataset จาก IG ที่ถูกแบ่งออกไป

auto_labeled_data.csv                 => dataset ที่ได้จาก auto_label.py   

auto_labeled_filtered.csv             => dataset ที่ผ่านการกรองด้วย filter_and_balance.py แล้ว

prediction_results_per_menu.csv       => คำตอบที่ได้จาก predict_per_menu_no_answer.py หรือ predict_per_menu.py

model_xxxxxxx.pth                     => model ที่ได้จาก train_per_menu.py
