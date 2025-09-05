@echo off
chcp 65001 >nul
echo ========================================
echo    A股智能选股系统 启动脚本
echo ========================================
echo.

:menu
echo 请选择要执行的操作：
echo 1. 训练模型
echo 2. 预测选股
echo 3. 策略回测
echo 4. 启动定时任务
echo 5. 测试推送渠道
echo 6. 退出
echo.
set /p choice=请输入选项数字 (1-6): 

if "%choice%"=="1" goto train
if "%choice%"=="2" goto predict
if "%choice%"=="3" goto backtest
if "%choice%"=="4" goto schedule
if "%choice%"=="5" goto test
if "%choice%"=="6" goto exit
echo 无效选项，请重新选择
goto menu

:train
echo.
echo 正在启动模型训练...
python main.py --mode train
pause
goto menu

:predict
echo.
echo 正在运行股票预测...
python main.py --mode predict
pause
goto menu

:backtest
echo.
echo 正在运行策略回测...
python main.py --mode backtest
pause
goto menu

:schedule
echo.
echo 正在启动定时任务调度器...
echo 按 Ctrl+C 可停止定时任务
python scheduler.py
pause
goto menu

:test
echo.
echo 正在测试推送渠道...
python -c "from src.notification import Notifier; from config.settings import load_config; n = Notifier(load_config()); print('测试结果:', n.test_all_channels())"
pause
goto menu

:exit
echo.
echo 谢谢使用！
pause
exit
