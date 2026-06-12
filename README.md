# 2026-1-LLM-Project
[2026-1] 고급비즈니스애널리틱스 LLM 프로젝트 3팀 레포지토리입니다.


# LLM 실행
```
# 패키지 설치
pip install openai

# 실행
python run.py --store "악어떡볶이"
python run.py --store "악어떡볶이" --owner "배달 매출을 늘리고 싶어요"
python run.py --store "악어떡볶이" --no-stream          # 스트리밍 없이 결과만 출력
python run.py --store "악어떡볶이" --save               # 리포트를 txt 파일로 저장
```

# Streamlit 실행
<img width="2522" height="1204" alt="image" src="https://github.com/user-attachments/assets/91cd663f-88ff-4610-bede-a35f4ba5b6d4" />

```
# 패키지 설치
pip install streamlit

# 실행
streamlit run app/streamlit_app.py
```
