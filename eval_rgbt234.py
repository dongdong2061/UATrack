from rgbt import RGBT234

rgbt234 = RGBT234()

# Register your tracker
# rgbt234(
#     tracker_name="tracker1",
#     result_path="./result/RGBT234/TFNet", 
#     bbox_type="corner",
#     prefix="TFNet_")

rgbt234(
    tracker_name="tracker2",
    result_path="/DATA/dingzhaodong/project/BAT2/BAT/RGBT_workspace/results/RGBT234/rgbt11", 
    bbox_type="ltwh"
)

# # Evaluate multiple trackers
pr_dict = rgbt234.MPR()
# print(pr_dict["tracker1"][0])
print(pr_dict["tracker2"][0])

# sr_dict = rgbt234.MSR()
# print(sr_dict["tracker2"][0])


# Evaluate single tracker
apf_sr,_ = rgbt234.MSR("tracker2")
print("APFNet MSR: \t", apf_sr)
# tf_sr,_ = rgbt234.MSR("TFNet")
# print("TFNet MSR: \t", tf_sr)
# jmmac_sr,_ = rgbt234.MSR("JMMAC")
# print("JMMAC MSR: \t", jmmac_sr)


# Evaluate single challenge
pr_tc_dict = rgbt234.MPR(seqs=rgbt234.TC)
sr_tc_dict = rgbt234.MSR(seqs=rgbt234.TC)

# Draw a radar chart of all challenge attributes
rgbt234.draw_attributeRadar(metric_fun=rgbt234.MPR, filename="RGBT234_MPR_radar.png")
rgbt234.draw_attributeRadar(metric_fun=rgbt234.MSR)     # this is ok

# Draw a curve plot
rgbt234.draw_plot(metric_fun=rgbt234.MPR)
rgbt234.draw_plot(metric_fun=rgbt234.MSR)