import Lucas_Kanade_Low_Level as lkll
import Lucas_Kanade_student_version as lksv
import Farneback as fb
import optical_flow_RAFT_model as raft
import cornerness as corn

# Execution Time Factor
def exe_factor(cost1, cost2):
    time_factor = cost1 / cost2
    return time_factor


def exersice2(video):
    method1 = "Lucas Kanade student version model"
    method2 = "Farneback model"
    lucas_student_cost = lksv.lucaskanadesv_method(video)
    farneback_cost = fb.fanerback_method(video)

    tfactor = exe_factor(lucas_student_cost, farneback_cost)

    return lucas_student_cost, farneback_cost, tfactor, method1, method2


def exersice3(video):
    method1 = "Optical Flow RAFT model"
    method2 = "Farneback model"
    raft_cost = raft.raft_method(video)
    farneback_cost = fb.fanerback_method(video)

    tfactor = exe_factor(raft_cost, farneback_cost)

    return raft_cost, farneback_cost, tfactor, method1, method2


def exersice4(video):
    method1 = "Farneback model"
    method2 = "Lucas Kanade Low Level model"
    farneback_cost = fb.fanerback_method(video)
    lucas_ll_cost = lkll.lkll_method(video)

    tfactor = exe_factor(farneback_cost, lucas_ll_cost)

    return farneback_cost, lucas_ll_cost, tfactor, method1, method2


def exersice5(video):
    corn.cornersMap(video)
    return
