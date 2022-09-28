---
title: "Setup a Raspberry Pi Local Network from Scratch"
date: 2022-09-28
categories: 
  - Engineering Pain
tags: 
  - edge device
header: 
  image: "/assets/images/setup-raspberry-pi4/pi-cover.jpg"
toc: true
toc_sticky: true
---

👋 Hi there. Welcome back to my page. Recently, I have practiced implementing a real-world [Federated Learning](https://en.wikipedia.org/wiki/Federated_learning) setting on edge devices, not just simulation on a single machine. At first, I had to create a local network of edge devices, [Raspberry Pi](https://www.raspberrypi.com/) computers in particular, via [Ethernet](https://en.wikipedia.org/wiki/Ethernet) connections. It took me one day and a half to config everything with a lot of pain, therefore, I decided to start a new category on my page, [Engineering Pain](https://gather-ai.github.io/categories/#engineering-pain), to write down and share engineering things like this. Hope you find them useful. 
{: style="text-align: justify;"}

## 1. Requirements
You need all of these things to start: 
{: style="text-align: justify;"}
* At least two Raspberry Pi, I use Pi 4 in my setting
* One micro SD card for each Pi
* One monitor, one mouse, and one keyboard
* An Ethernet switch
* A bunch of cables of various types
{: style="text-align: justify;"}

## 2. Boot OS into Pi
Firstly, we need to install Raspbian OS on X by following steps: 
{: style="text-align: justify;"}
* Plug your SD card into your computer (maybe via a reader USB)
{: style="text-align: justify;"}
* Download and use the [SD card formatter](https://www.sdcard.org/downloads/formatter/) tool to format your card, this process will take a while, depending on card memory size
{: style="text-align: justify;"}
<figure class="align-center" style="width: 400px">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/setup-raspberry-pi4/sd-card-formatter.jpg">
</figure>
* Download and unzip [Raspbian](https://downloads.raspberrypi.org/raspbian_full_latest) from this link
{: style="text-align: justify;"}
* Download and use [Win32DiskImager](https://sourceforge.net/projects/win32diskimager/) to write OS into SD card
{: style="text-align: justify;"}
<figure class="align-center" style="width: 400px">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/setup-raspberry-pi4/win32diskimager.jpg">
</figure>
* Put SD card into Pi and turn it on, plug in a monitor, mouse, and keyboard, then do some simple steps to start working
{: style="text-align: justify;"}

## 3. Enable SSH via Ethernet
Now that Pi is ready, we need to connect to it from our main computer. One option is to use SSH via the Ethernet cable. 
{: style="text-align: justify;"}
* On your Pi, go to the Pi icon &rarr; Preferences &rarr; Raspberry Pi Configuration
{: style="text-align: justify;"}
<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/setup-raspberry-pi4/pi-configuration.jpg">
</figure>
* Click on the Interfaces tab &rarr; Check the Enabled button for SSH
{: style="text-align: justify;"}
<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/setup-raspberry-pi4/pi-configuration-interfaces.jpg">
</figure>
* Open Terminal, use the command line `ifconfig` to get the IP address of your Pi
{: style="text-align: justify;"}
* Connect the Ethernet cable from Pi to your computer, then use any SSH client (PuTTY for example) to connect to Pi by its IP address
{: style="text-align: justify;"}

## 4. Share Internet connection from your computer
Next, we need the Internet connection on Pi to install a lot of Python packages for working on Federated Learning. 
{: style="text-align: justify;"}
* Open Control Panel &rarr; Network and Internet &rarr; Network and Sharing Center, where you see the Connections fields for Wi-Fi and Ethernet
{: style="text-align: justify;"}
<figure class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/setup-raspberry-pi4/network-and-sharing-center.jpg">
</figure>
* Under the Connections field for Wi-Fi, click on the name of your Wi-Fi connection to open the Wi-Fi Status window
{: style="text-align: justify;"}
<figure class="align-center" style="width: 300px">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/setup-raspberry-pi4/wi-fi-status.jpg">
</figure>
* Open Properties &rarr; click on Sharing tab &rarr; Check the box for “Allow other network users to connect through this computer’s Internet connection” &rarr; select the Ethernet connection in the drop-down menu below
{: style="text-align: justify;"}
<figure class="align-center" style="width: 300px">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/setup-raspberry-pi4/wi-fi-properties.jpg">
</figure>

Now, our Pi can access the Internet shared from our computer, we can now create a virtual environment and install any Python package into it. 
{: style="text-align: justify;"}

## 5. Broaden the local network
Final and easiest step, after completing all the above steps, we can create a wider local network by using an Ethernet switch. Simply, connect all Ethernet cables from the main computer and all of Pi to the switch, no further configuring is required. 
{: style="text-align: justify;"}

Stay tuned for more content!
{: style="text-align: justify;"}

## References
[[1] Computer Vision & Pi – Chương 1. Cài đặt môi trường lập trình cho Raspberry Pi](https://miai.vn/2020/02/17/computer-vision-pi-chuong-1-cai-dat-moi-truong-lap-trinh-cho-raspbery-pi/)<br>
[[2] First Steps with your Raspberry Pi](https://medium.com/secure-and-private-ai-writing-challenge/first-steps-with-your-raspberry-pi-5917f980a48)<br>
[[3] Connecting to your Raspberry Pi via SSH](https://raspberrypi-guide.github.io/networking/connecting-via-ssh)<br>
[[4] How to Use Internet Connection Sharing on Windows 10](https://medium.com/@jrcharney/connect-your-raspberry-pi-to-your-computer-via-ethernet-4564e1e68922)<br>
{: style="text-align: justify;"}