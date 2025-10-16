
import requests
import xlrd
import xlwt
import time
import random


class ArticleArchiveDoi:
    def __init__(self, filename, url_publisher, apikey, arformat, path, text_outpath):
        self.filename = filename
        self.url_publisher = url_publisher
        self.apikey = apikey
        self.arformat = arformat
        self.path = path
        self.out_path = text_outpath

    def data_totxt(self, sample, path):
        f = open(path, 'w', encoding='utf-8')
        f.write(sample)
        f.close()

    def httprequest(self):
        proxyList = ['http://112.51.96.118:9091', 'http://113.200.56.13:8010', 'http://119.28.128.206:9000',
         'http://180.121.135.177:8089', 'http://114.232.109.215:8089', 'http://120.194.189.122:8908',
         'http://106.75.140.91:8888', 'http://120.194.18.90:81', 'http://119.28.138.104:3128', 'http://121.8.98.198:80',
         'http://59.44.16.6:80', 'http://182.106.220.252:9091', 'http://117.160.250.133:80', 'http://47.98.49.223:3128',
         'http://120.132.52.250:8888', 'http://39.165.0.137:9002', 'http://218.56.132.158:8080',
         'http://114.231.45.162:8089', 'http://183.165.250.112:8089', 'http://193.112.1.24:3128',
         'http://118.190.210.227:3128', 'http://43.247.68.211:3128', 'http://114.232.110.177:8089',
         'http://114.215.102.168:8081', 'http://221.202.248.52:80', 'http://139.129.166.68:3128',
         'http://117.57.93.221:8089', 'http://60.205.228.133:9999', 'http://120.132.52.97:8888',
         'http://180.123.193.157:8089', 'http://180.121.133.123:8089', 'http://121.8.98.197:80',
         'http://171.34.197.71:3128', 'http://140.143.96.216:80', 'http://111.62.243.64:80', 'http://47.100.91.57:8080',
         'http://110.81.178.254:8899', 'http://19.28.50.37:82', 'http://122.72.18.34:80', 'http://122.72.18.35:80',
         'http://117.158.220.89:8908', 'http://114.232.110.100:8089', 'http://140.143.134.248:3128',
         'http://222.168.111.178:8060', 'http://106.75.135.252:8888', 'http://222.33.192.238:8118',
         'http://118.24.61.22:3128', 'http://27.36.116.226:3128', 'http://14.29.47.90:3128',
         'http://180.123.195.212:8089', 'http://47.98.119.145:3128', 'http://120.132.52.16:8888',
         'http://221.224.49.237:3128', 'http://47.94.230.42:9999', 'http://222.73.68.144:8090',
         'http://112.24.107.102:8908', 'http://47.243.114.192:8180', 'http://58.22.248.12:8908',
         'http://124.133.230.254:80', 'http://119.28.26.57:3128', 'http://118.120.230.107:41122',
         'http://180.150.191.153:8888', 'http://112.115.57.20:3128', 'http://119.27.177.169:80',
         'http://101.4.136.34:81', 'http://119.28.112.130:3128', 'http://119.28.221.28:8088',
         'http://117.69.232.91:8089', 'http://180.120.215.175:8089', 'http://203.74.125.18:8888',
         'http://183.166.170.136:41122']

        USER_AGENTS = [
            'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.4295.400',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36 OPR/26.0.1656.60',
            'Opera/8.0 (Windows NT 5.1; U; en)',
            'Mozilla/5.0 (Windows NT 5.1; U; en; rv:1.8.1) Gecko/20061208 Firefox/2.0.0 Opera 9.50',
            'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; en) Opera 9.50',
            'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0',
            'Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
            'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 Safari/534.16',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 TaoBrowser/2.0 Safari/536.11',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER',
            'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; LBBROWSER)',
            'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 SE 2.X MetaSr 1.0',
            'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; SE 2.X MetaSr 1.0)',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.122 UBrowser/4.0.3214.0 Safari/537.36'
            ]
        xls = xlwt.Workbook()
        sheet = xls.add_sheet("doi-text_path")
        data = xlrd.open_workbook(self.filename)
        table = data.sheet_by_index(0)
        list_values = []
        for x in range(1, len(table.col_values(1))):
            values = []
            row = table.row_values(x)
            values.append(row[1])
            list_values.append(values)
        dois = list_values
        print(dois)
        count = len(dois)
        for i in range(0, count):      # 修改
            url = self.url_publisher + dois[i][0] + "?" + self.apikey + "&httpAccept=" + self.arformat
            # ***************************************************
            # 设置随机header和proxies
            try:
                user_agent = random.choice(USER_AGENTS)
                header = {"Accept": "text/plain", "CR-TDM-Rate-Limit": "4000", "CR-TDM-Rate-Limit-Remaining": "76",
                           "CR-TDM-Rate-Limit-Reset": "1378072800",
                           'User-Agent': user_agent}  # 定义http请求的参数
                proxies = {"http": random.choice(proxyList)}
                # ***************************************************
                r = requests.get(url, headers=header, proxies=proxies)
                content = r.content.decode()
                time.sleep(1)  # 等待1秒
                print(self.path)
                path = self.path + '/' + str(i) + '.txt'
                print(path)
                self.data_totxt(content, path)
                sheet.write(i, 0, dois[i][0])
                sheet.write(i, 1, path)
            except:
                user_agent = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.4295.400'
                header = {"Accept": "text/plain", "CR-TDM-Rate-Limit": "4000", "CR-TDM-Rate-Limit-Remaining": "76",
                  "CR-TDM-Rate-Limit-Reset": "1378072800",
                  'User-Agent': user_agent}  # 定义http请求的参数
                proxies = {"http": 'http://59.44.16.6:80', }
                r = requests.get(url, headers=header, proxies=proxies)
                content = r.content.decode()
                time.sleep(1)  # 等待1秒
                print(self.path)
                path = self.path + '/' + str(i) + '.txt'
                print(path)
                self.data_totxt(content, path)
                sheet.write(i, 0, dois[i][0])
                sheet.write(i, 1, path)

        xls.save(self.out_path)