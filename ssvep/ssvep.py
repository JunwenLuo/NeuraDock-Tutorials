from psychopy import visual, core, event, logging
import os

# ================= 配置区域 =================
# 屏幕参数
SCREEN_REFRESH_RATE = 60  # 请务必设置成你实际显示器的刷新率
STIM_FREQUENCY = 4      # 目标刺激频率 (Hz)
# 计算每周期多少帧 (比如 60Hz 屏幕做 10Hz 刺激，就是每 6 帧一个周期)
# 注意：最理想的情况是 (刷新率 / 频率) 是一个整数
FRAMES_PER_CYCLE = int(SCREEN_REFRESH_RATE / STIM_FREQUENCY)

# 实验参数
TRIAL_NUM = 6            # 试次数量 (一共闪烁几次)
STIM_DURATION = 1.0       # 每次刺激持续时间 (秒)
REST_DURATION = 1.0       # 每次刺激中间的休息时间 (秒)

# 文件路径
MARKER_FILE = '1.txt'
IMG_TARGET = 'white.png'
IMG_BG = 'black.png'
# ===========================================

def write_marker():
    """
    向 1.txt 文件中写入 marker
    使用 'a' (append) 模式，以免覆盖之前的记录
    """
    try:
        with open(MARKER_FILE, 'a') as f:
            f.write('marker\n')
        print(f"Marker written to {MARKER_FILE}")
    except Exception as e:
        print(f"Error writing marker: {e}")

def run_experiment():
    # 1. 创建窗口 (fullscr=True 为全屏，测试时可改为 False)
    win = visual.Window(
        size=[900, 900],
        fullscr=True, 
        screen=0, 
        units='pix',
        color='black'
    )

    # 2. 加载图片刺激
    # background (black.png) 通常作为底色或者 Off 状态
    stim_off = visual.ImageStim(win, image=IMG_BG,size=[400, 400])
    # target (white.png) 作为 On 状态
    stim_on = visual.ImageStim(win, image=IMG_TARGET, size=[400, 400])
    
    # 指导语
    text_instr = visual.TextStim(win, text='按空格键开始实验...', color='white')
    text_instr.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

    # 3. 实验循环
    for i in range(TRIAL_NUM):
        # ---------------- 休息阶段 ----------------
        rest_text = visual.TextStim(win, text=f'Rest ({i+1}/{TRIAL_NUM})', color='grey')
        rest_text.draw()
        win.flip()
        core.wait(REST_DURATION)

        # ---------------- 刺激阶段 ----------------
        
        # !! 关键步骤：在刺激即将开始前写入 Marker !!
        

        # 计算总共需要翻转多少帧
        total_frames = int(STIM_DURATION * SCREEN_REFRESH_RATE)

        write_marker()
        for frame_n in range(total_frames):
            # SSVEP 逻辑 (方波闪烁):
            # 这里的逻辑是：周期的前半段显示 White，后半段显示 Black
            # 举例 60Hz 10Hz: 周期6帧。0,1,2 显示 White; 3,4,5 显示 Black
            
            phase = frame_n % FRAMES_PER_CYCLE
            if phase < (FRAMES_PER_CYCLE / 2):
                stim_on.draw()  # 显示 white.png
            else:
                stim_off.draw() # 显示 black.png (或者不画，即显示背景)
            
            win.flip() # 刷新屏幕
            
            # 允许按 ESC 退出
            if 'escape' in event.getKeys():
                win.close()
                core.quit()

    # 结束
    end_text = visual.TextStim(win, text='实验结束', color='white')
    end_text.draw()
    win.flip()
    core.wait(2)
    
    win.close()
    core.quit()

if __name__ == "__main__":
    # 确保之前的文件被清空（可选，如果你希望每次运行都重置文件）
    with open(MARKER_FILE, 'w') as f:
        f.write("HEADER_DEF,_,C,C,C,C,C,C,C\n")
        
    run_experiment()