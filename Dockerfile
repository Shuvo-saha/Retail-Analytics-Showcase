FROM python:3.6.13
COPY . /app
COPY requirements.txt ./requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["streamlit_app.py"]