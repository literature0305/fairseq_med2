#!/usr/bin/env bash

./8_decode_WMT14en2de.sh &> errlog007-2_decode_best
./8_decode_WMT14en2de_avg.sh 95 &> errlog007-2_decode_wmt14en2de_avg_86-95
./8_decode_WMT14en2de_avg.sh 75 &> errlog007-2_decode_wmt14en2de_avg_66-75
./8_decode_WMT14en2de_avg.sh 65 &> errlog007-2_decode_wmt14en2de_avg_56-65
./8_decode_WMT14en2de_avg.sh 60 &> errlog007-2_decode_wmt14en2de_avg_51-60
./8_decode_WMT14en2de_avg.sh 55 &> errlog007-2_decode_wmt14en2de_avg_46-55
./8_decode_WMT14en2de_avg.sh 50 &> errlog007-2_decode_wmt14en2de_avg_41-50
./8_decode_WMT14en2de_avg.sh 45 &> errlog007-2_decode_wmt14en2de_avg_36-45
./8_decode_WMT14en2de_avg.sh 40 &> errlog007-2_decode_wmt14en2de_avg_31-40
