import streamlit as st
import requests
import base64
import plotly.graph_objects as go
import mediapipe as mp

# 페이지 기본 설정
st.set_page_config(layout="wide", page_title="Human Pose 3D Estimation")

st.title("🏃‍♂️ Human Pose 2D & 3D Estimation UI")
st.markdown("FastAPI 백엔드로 사진을 전송하고, 반환된 **2D 스켈레톤 이미지**와 **3D 관절 좌표**를 Plotly로 함께 확인하세요.")

uploaded_file = st.file_uploader("사람이 포함된 이미지를 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="원본 이미지", width=300)
    
    if st.button("AI 분석 시작!"):
        with st.spinner("FastAPI 서버에서 MediaPipe 분석 중입니다..."):
            # FastAPI 예측 엔드포인트
            url = "http://localhost:8000/predict_full"
            try:
                res = requests.post(url, files={"file": uploaded_file.getvalue()})
                if res.status_code == 200:
                    data = res.json()
                    
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        st.success("관절 및 3D 좌표 추출 성공!")
                        col1, col2 = st.columns(2)
                        
                        # 1. 2D Annotated Image Display
                        with col1:
                            st.subheader("🖼️ 2D 스켈레톤 오버레이")
                            image_bytes = base64.b64decode(data["image_base64"])
                            st.image(image_bytes, use_container_width=True)
                        
                        # 2. 3D Skeleton UI Display using Plotly
                        with col2:
                            st.subheader("🌐 3D 스켈레톤 모델 (Plotly)")
                            landmarks = data.get("world_landmarks", [])
                            if landmarks:
                                # 축 매핑 (MediaPipe World Landmarks는 x, y, z로 주어짐)
                                # Plotly의 3D 공간에 사람이 서있는 형태로 자연스럽게 매핑하기 위해 축을 회전합니다.
                                x = [lm["x"] for lm in landmarks]
                                y = [-lm["y"] for lm in landmarks]  # Y축 반전
                                z = [lm["z"] for lm in landmarks]
                                
                                fig = go.Figure()
                                
                                # 매핑: X -> 좌우, Y -> 깊이(원근), Z -> 상하(키)
                                px = x
                                py = z
                                pz = y
                                
                                mp_pose = mp.solutions.pose
                                
                                # 뼈대 연결선 그리기
                                for connection in mp_pose.POSE_CONNECTIONS:
                                    start_idx = connection[0]
                                    end_idx = connection[1]
                                    
                                    fig.add_trace(go.Scatter3d(
                                        x=[px[start_idx], px[end_idx]],
                                        y=[py[start_idx], py[end_idx]],
                                        z=[pz[start_idx], pz[end_idx]],
                                        mode='lines',
                                        line=dict(color='orange', width=5),
                                        showlegend=False
                                    ))
                                
                                # 관절 포인트 그리기
                                fig.add_trace(go.Scatter3d(
                                    x=px, y=py, z=pz,
                                    mode='markers',
                                    marker=dict(size=6, color='cyan', line=dict(width=1, color='darkblue')),
                                    showlegend=False
                                ))
                                
                                # 레이아웃 설정 (축 비율 맞추기)
                                fig.update_layout(
                                    scene=dict(
                                        xaxis=dict(title="X (좌우)", showticklabels=False),
                                        yaxis=dict(title="Z (깊이)", showticklabels=False),
                                        zaxis=dict(title="Y (높이)", showticklabels=False),
                                        aspectmode='data'
                                    ),
                                    margin=dict(l=0, r=0, b=0, t=0),
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("사람을 찾지 못해 3D 관절 좌표를 추출할 수 없습니다.")
            except requests.exceptions.ConnectionError:
                st.error("오류: 서버에 연결할 수 없습니다. 터미널에서 FastAPI 서버(8000포트)가 켜져 있는지 확인해주세요!")
