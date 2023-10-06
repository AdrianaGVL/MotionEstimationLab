import functions as funct
import Lucas_Kanade_Low_Level as lkll
import Lucas_Kanade_student_version as lksv
import Farneback as fb
import optical_flow_RAFT_model as raft

# Parameters
video = 'traffic.mp4'

# print("Which Exersice from Lab 1 you want to execute?")
# print("\t Exersice 2: Lucas_Kanade_student_version vs. Farneback?")
# print("\t Exersice 3: optical_flow_RAFT_model vs. Farneback?")
# print("\t Exersice 4: Farneback vs. Lucas_Kanade_Low_Level?")
# print("\t Exersice 5: Cornerness response map  in each frame")
# exersice = input("Write the number of exersice you want to execute and then press enter: \n")
# exersice = int(exersice)
#
# if not isinstance(exersice, int) or (6 <= exersice or exersice <= 1):
#     print("The value isn't valid \n")
#
# else:
#     match exersice:
#         case 2:
#             time1, time2, time_factor, method1, method2 = funct.exersice2(video)
#         case 3:
#             time1, time2, time_factor, method1, method2 = funct.exersice3(video)
#         case 4:
#             time1, time2, time_factor, method1, method2 = funct.exersice4(video)
#         case 5:
#             funct.exersice5(video)
#         case _:
#             print("Error")
#
#     if 5 > exersice > 1:
#         print(f"Computational cost from {method1} was: {time1} \n"
#               f"Computational cost from {method2} was: {time2} \n"
#               f"The final execution time factor was {time_factor}")