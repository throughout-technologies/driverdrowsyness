import cv2
import mediapipe as mp
from playsound import playsound
import sendemail

mp_facemesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
cordinate = mp_drawing._normalized_to_pixel_coordinates


choosen_left_eye = [362,385,387,263,373,380]
choosen_right_eye = [33,160,158,133,153,144]

all_choosen_idx = choosen_left_eye + choosen_right_eye


cap=cv2.VideoCapture(0)
with mp_facemesh.FaceMesh(
  static_image_mode =False,
  max_num_faces = 1,
  refine_landmarks=False,
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5
) as face_mesh:

  def distance(point_1, point_2):
      """Calculate l2-norm between two points"""
      
      dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
      return dist
    
  def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = cordinate(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)
        breakpoint()
        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])
 
        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
 
    except:
        ear = 0.0
        coords_points = None
 
    return ear, coords_points
  
  def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    """Calculate Eye aspect ratio"""
 
    left_ear, left_lm_coordinates = get_ear(
                                      landmarks, 
                                      left_eye_idxs, 
                                      image_w, 
                                      image_h
                                    )
    right_ear, right_lm_coordinates = get_ear(
                                      landmarks, 
                                      right_eye_idxs, 
                                      image_w, 
                                      image_h
                                    )
    Avg_EAR = (left_ear + right_ear) / 2.0
 
    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)

  while True:

    rel,frames=cap.read()
    frames = cv2.cvtColor(frames,cv2.COLOR_BGR2RGB)
    imgH,imgW,_=frames.shape
    result = face_mesh.process(frames)
    
    frames = cv2.cvtColor(frames,cv2.COLOR_RGB2BGR)
    
    if result.multi_face_landmarks:
      
      for face_landmarks in result.multi_face_landmarks:
        landmarks = face_landmarks.landmark
        
        for landmark_idx, landmark in enumerate(landmarks):
          if landmark_idx in all_choosen_idx:
            pred_cord = cordinate(landmark.x, landmark.y,imgW, imgH)
            cv2.circle(frames, pred_cord,2,(0, 255, 255),cv2.FILLED)
            
            EAR, _ = calculate_avg_ear(
                          landmarks, 
                          choosen_left_eye, 
                          choosen_right_eye, 
                          imgW, 
                          imgH
                      )
 
                # Print the EAR value on the custom_chosen_lmk_image.
            cv2.putText(frames,f"EAR: {round(EAR, 2)}", (1, 24), cv2.FONT_HERSHEY_COMPLEX,  0.9, (255, 255, 0), 2 )                
            if EAR < 0.15 :
              cv2.putText(frames,"sleep",(2,100),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,0,255), 2)
              sendemail.mail()
              break
              # playsound('/path/note.wav')
              
            elif EAR > 0.15 and EAR < 0.24 :
              cv2.putText(frames,"drowsy",(2,100),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,0,255,), 2)
            else :
              cv2.putText(frames,"activee!",(2,100),cv2.FONT_HERSHEY_COMPLEX,0.9,(0, 255, 0), 2)
    cv2.imshow("video",frames)
    if cv2.waitKey(2)==27:
        break
cap.release()
cv2.destroyAllWindows()
