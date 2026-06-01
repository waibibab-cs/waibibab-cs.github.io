# Claude code
## 配置相关
官网订阅claude code是一月25美元，且听说貌似还有限额，但某鱼代充的话感觉有很大的封号风险，暂时没有找到一个特别好的方案。目前仅尝试了在某宝上开通的国产平台api key的体验卡，8元50次提问，国产这些平台有一个很抽象的地方是总是按照提问次数收费而不是token数，包括之前在simplex那个网站上使用gemini flash也是这样。

具体配置方法也是按照客服给的教程，非常简单全程没有出现像geminicli那样非常多的bug
## 使用体验
初步感觉功能确实强大，不过不知道是不是由于不是官网的原因，经常发生interrupted的崩溃，目前已经提问了30次左右，有一半都是提问失败的。
目前还未弄清楚interrupted的规律，刚开始以为是问题太复杂导致的，毕竟崩溃基本都是发生在它在自动检索分析代码的过程中，但有时候又能维持检索5分钟不崩溃，可能也与使用时间有关，后面尽量避开高峰期试试。
# gemini-cli
## 配置相关
配置除了官网给的教程，还配了代理，可以参考：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260318231659.png)
其中7890是vpn（如clash右上角有写）的端口号
必须打开clash的“允许局域网接入clash”
此外可以通过以下指令预先测试是否连通google
```bash
curl -I https://www.google.com
```
此外，key认证如果是免费的，限额很严重，建议登录认证。
## 使用体验
唯一的好处是一天可以发送1000个请求，不过实际上一次涉及代码搜索的提问一般都会包含二三十个请求，所以实际上一天真正可以进行的大提问有30+次。
不过效果的话就比较一般，同一个代码分析问题先后提问了geminicli和claude，前者仅生成了50行的报告，写得比较笼统，后者报告长达600行，非常详细，如下图所示：
![image.png](https://raw.githubusercontent.com/waibibab-cs/blog_img/main/cdnimg/20260319100301.png)
