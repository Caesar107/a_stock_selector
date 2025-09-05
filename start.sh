#!/bin/bash

echo "========================================"
echo "    A股智能选股系统 启动脚本"
echo "========================================"
echo

show_menu() {
    echo "请选择要执行的操作："
    echo "1. 训练模型"
    echo "2. 预测选股"
    echo "3. 策略回测"
    echo "4. 启动定时任务"
    echo "5. 测试推送渠道"
    echo "6. 退出"
    echo
}

while true; do
    show_menu
    read -p "请输入选项数字 (1-6): " choice
    
    case $choice in
        1)
            echo
            echo "正在启动模型训练..."
            python main.py --mode train
            read -p "按回车键继续..."
            ;;
        2)
            echo
            echo "正在运行股票预测..."
            python main.py --mode predict
            read -p "按回车键继续..."
            ;;
        3)
            echo
            echo "正在运行策略回测..."
            python main.py --mode backtest
            read -p "按回车键继续..."
            ;;
        4)
            echo
            echo "正在启动定时任务调度器..."
            echo "按 Ctrl+C 可停止定时任务"
            python scheduler.py
            ;;
        5)
            echo
            echo "正在测试推送渠道..."
            python -c "from src.notification import Notifier; from config.settings import load_config; n = Notifier(load_config()); print('测试结果:', n.test_all_channels())"
            read -p "按回车键继续..."
            ;;
        6)
            echo
            echo "谢谢使用！"
            exit 0
            ;;
        *)
            echo "无效选项，请重新选择"
            ;;
    esac
    echo
done
