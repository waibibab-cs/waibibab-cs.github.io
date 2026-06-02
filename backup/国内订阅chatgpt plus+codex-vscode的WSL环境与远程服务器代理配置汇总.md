# 前言
本文汇总教程，展示如何在国内订阅官方版chatgpt plus、同时在本地wsl2环境与远程服务器的环境下的vscode上配置codex的代理。
苹果安卓都是可以的，苹果从step1开始看，安卓直接跳转step2看教程，注:chatgpt plus价格20美元
# step1 注册一个美区apple id
直接参考教程：https://zhuanlan.zhihu.com/p/1967272520882300209
非常详细
# step2 美区app store下载chatgpt并订阅
直接参考教程：https://monkeywie.cn/posts/chatgpt-plus-subscription-via-app-sotre-google-play
非常详细，不过其中pockytshop我在支付宝没有搜到，解决办法是直接在网页上搜，然后用手机支付宝登录并支付，但是当我付款之后的瞬间就被拒绝，同时接到了一个支付宝客服的电话（机器人）进行了风险提示，问是否本人操作之类的，不过在支付宝上跟着流程操作解禁之后就恢复并可以成功购买了
# step3 如何在WSL-vscode上配置代理
如果直接下载codex插件，并尝试使用chatgpt账号登录必会失败，这时候需要配置代理（注意结点最好别用香港，我用的日本），在wsl上配置代理的方法其实没有什么特殊的：

步骤 1: 获取 Windows 的 IP 地址
- 在 WSL2 中运行以下命令，查看 DNS 服务器的 IP 地址： `cat /etc/resolv.conf` 通常会显示类似 nameserver 172.xx.xx.xx 的地址，这个地址即为 Windows 的 IP
- 可以理解为Windows是一个系统、WSL是另一个独立的系统，而127.0.0.1是虽然在Windows的视角中是windows本地的ip，但在WSL的视角中是WSL本地的ip，而我们的代理软件如clash位置在windows而非wsl里，所以如果使用127.0.0.1:7890访问的是wsl里的clash，根本没有这个东西，而172.xx.xx.xx就是wsl访问windows的ip
步骤 2: 配置环境变量
- 编辑 `~/.bashrc` 文件：
vim ~/.bashrc
- 添加以下两行内容（根据实际情况替换 IP 和端口）： 
- export http_proxy="http://172.xx.xx.xx:7890" 
- export https_proxy="http://172.xx.xx.xx:7890" 
- IP：填写上一步获取的 Windows IP 地址。 
- 端口：填写本地代理客户端的端口（如 clash是7890）。
- 保存并刷新配置：
source ~/.bashrc

步骤 3: 配置 Windows 本地代理客户端
1. 确保你的代理客户端（如 Clash、V2Ray 等）已开启，并允许来自局域网的请求。
2. 在代理客户端中启用“允许局域网连接”选项，并确认端口号与上述配置一致。

步骤 4: 验证代理是否生效
- 在 WSL2 中尝试访问 Google：
- 如果返回正常的 HTTP 响应，则说明代理配置成功。

**提示与注意事项**
- 如果需要临时设置代理，可以直接在终端中运行：
export http_proxy="http://172.xx.xx.xx:7890"
export https_proxy="http://172.xx.xx.xx:7890"
- 若不再需要代理，可通过以下命令清除环境变量：
unset http_proxy
unset https_proxy
# step4 如何在VScode上配置远程服务器代理
这里要解决的问题是：本地电脑可以通过代理访问 Codex，但是远程服务器因为网络限制，不能直接访问 `chatgpt.com` / OpenAI 相关服务，所以需要将远程服务器的请求走本地代理从而访问外网

可以直接参考教程：https://ultrafish.io/post/codex-vscode-remote-ssh-proxy/

这个教程对步骤和原理的讲解都十分详细