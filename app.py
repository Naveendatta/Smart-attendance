from flask import Flask, render_template, request
import os
import pandas as pd
import datetime
import time
import cv2
import numpy as np
import csv
from PIL import Image, ImageTk
from dataset import create_dataset

from recognition import Recognise

app = Flask(__name__)

@app.route('/')
def index():
   return render_template("index.html")



@app.route('/create_datsets',  methods=['POST','GET'])
def create_datsets():
   if request.method == 'POST':
      Id = request.form['Id']
      Name = request.form['Name']
      Phone = request.form['Phone']
      Email = request.form['Email']
      Sem = request.form['Sem']
      Cource = request.form['Cource']
      Branch = request.form['Branch']
      usn = request.form['usn']

      print(Id+' '+Name+' '+Phone+' '+Email+' '+Sem+' '+Cource+' '+Branch+' '+usn)

      create_dataset(Name)
      
      msg = ['Images Saved for',
            'ID : ' + Id,
            'Name : ' + Name,
            'nPhone : ' + Phone,
            'Email : ' + Email,
            'Semester : ' + Sem,
            'Cource : ' + Cource,
            'Branch : ' + Branch,
            'Usn : ' + usn]
      
      row = [Id, Name, Phone, Email, Sem, Cource, Branch, usn]

      if not os.path.exists('StudentDetails.csv'):
         row1 = ['Id', 'Name', 'Phone', 'Email', 'Sem', 'Cource', 'Branch', 'usn']
         with open('StudentDetails.csv','w',newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row1)
         csvFile.close()

      with open('StudentDetails.csv','a', newline='') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()

      return render_template("index.html", msg=msg)
   return render_template("index.html")

@app.route('/image_attendance', methods=['GET', 'POST'])
def image_attendance():
   if request.method == 'POST':
      Subject = request.form['Subject']
      file_name = request.form['filename']
      # Check if the file has a .mp4 extension
      if file_name.lower().endswith('.mp4'):
         # Proceed with further logic
         print("The file is an MP4.")
         from test_video import RecogniseVideo
         num = RecogniseVideo('static/test/'+file_name)
         if num == 'Unknown':
            return render_template("index.html", msg = ['Unknown person'])
         else:
            attendence_info=[]
            df = pd.read_csv('StudentDetails.csv')

            col_names =  ['Name','Date','Time']
            attendance = pd.DataFrame(columns = col_names)
            
            ts = time.time()      
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            Hour,Minute,Second=timeStamp.split(":")
            attendance.loc[len(attendance)] = [num,date,timeStamp]
            fileName="StudentAttendence/"+str(Subject)+"/Attendence_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
            attendance.to_csv(fileName, index=False)
            for i in num:
               attendence_info.append(f"{i} is present")
            print("\n\n\n\n entered video \n\n\n")
            print(f"\n\n\n\n {attendence_info} \n\n\n")
            return render_template('index.html', image1='static/test/'+file_name, image2='static/test/out.jpg',
                     List=attendence_info,  subject=Subject, date=date, time=timeStamp)
      else:
         print("The file is not an MP4.")
         from recognition_image import Recognise
         num = Recognise('static/test/'+file_name)
         if num == 'Unknown':
            return render_template("index.html", msg = ['Unknown person'])
         else:
            df = pd.read_csv('StudentDetails.csv')

            col_names =  ['Name','Date','Time']
            attendance = pd.DataFrame(columns = col_names)
            
            ts = time.time()      
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            Hour,Minute,Second=timeStamp.split(":")
            attendance.loc[len(attendance)] = [num,date,timeStamp]
            fileName="StudentAttendence/"+str(Subject)+"/Attendence_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
            attendance.to_csv(fileName, index=False)

            attendence_info = [str(num) + ' is present']
            return render_template('index.html', image1='static/test/'+file_name, image2='static/test/out.jpg',
                     List=attendence_info,  subject=Subject, date=date, time=timeStamp)
   return render_template('index.html')

@app.route('/attendence',  methods=['POST','GET'])
def attendence():
   if request.method == 'POST':
      Subject = request.form['Subject']
      num = Recognise()
      if num == 'Unknown':
         return render_template("index.html", msg = ['Unknown person'])
      else:
         df = pd.read_csv('StudentDetails.csv')

         col_names =  ['Name','Date','Time']
         attendance = pd.DataFrame(columns = col_names)
        
         ts = time.time()      
         date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
         timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
         Hour,Minute,Second=timeStamp.split(":")
         attendance.loc[len(attendance)] = [num,date,timeStamp]
         fileName="StudentAttendence/"+str(Subject)+"/Attendence_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
         attendance.to_csv(fileName, index=False)

         attendence_info = [str(num) + ' is present']

         return render_template("index.html", List=attendence_info,  subject=Subject, date=date, time=timeStamp)
   return render_template("index.html")



if __name__ == "__main__":
   app.run(debug=True, use_reloader=False)
