#-*- coding:UTF-8 -*-
'''
Created on 2017-4-23

@author: Bo Xu <mailto:bolang1988@gmail.com>

@version: 1.0

@summary: 

'''
import re
import time
import logging


class CONVERT_DATE():
    
    zh2num_dict = None
    
    ymd_relist = None
    ym_relist = None
    md_relist = None
    y_relist = None
    


    
    def __init__(self):
        self.zh2num_dict = {u'一':'1',u'二':'2',u'三':'3',u'四':'4',u'五':'5',u'六':'6',u'七':'7',u'八':'8',u'九':'9',u'〇':'0',u'○':'0',u'O':'0',u'零':'0'}
        
        #Patterns抽取年、月、日
        ymd_re1 = re.compile(r'([\d]{3,4})[^\d]*-[^\d]*([\d]{1,2})[^\d]*-[^\d]*([\d]{1,2})')
        ymd_re2 = re.compile(r'([\d]{3,4})[^\d]*\.[^\d]*([\d]{1,2})[^\d]*\.[^\d]*([\d]{1,2})')
        ymd_re3 = re.compile(u'([\d]{3,4})[^\d]*年[^\d]*([\d]{1,2})[^\d]*月[^\d]*([\d]{1,2})[^\d]*')
        ymd_re4 = re.compile(r'^([\d]{3,4})/([\d]{1,2})/([\d]{1,2})$')
        ymd_re5 = re.compile(r'([\d]{4})([\d]{2})([\d]{2})')
        
        self.ymd_relist = [ymd_re1, ymd_re2, ymd_re3, ymd_re4, ymd_re5]
        
        #Patterns抽取年、月
        ym_re1 = re.compile(u'([\d]{3,4})[^\d]*年[^\d]*([\d]{1,2})[^\d]*月')
        ym_re2 = re.compile(r'([\d]{3,4})-([\d]{1,2})')
        ym_re3 = re.compile(r'([\d]{3,4})\.([\d]{1,2})')
        ym_re4 = re.compile(r'^([\d]{3,4})-([\d]{1,2})-00$')
        self.ym_relist = [ym_re1, ym_re2, ym_re3, ym_re4]

        
        #Patterns抽取月、日
        md_re1 = re.compile(u'([\d]{1,2})[^\d]*月[^\d]*([\d]{1,2})[^\d]*[日,号]')
        md_re2 = re.compile(r'^0000-([\d]{2})-([\d]{2})$')
        self.md_relist = [md_re1, md_re2]
        
        #Patterns抽取年
        y_re1 = re.compile(u'([\d]{1,4})年')
        y_re2 = re.compile(u'^([\d]{3,4})$')
        y_re3 = re.compile(r'^([\d]{4})-00-00$')
        

        self.y_relist = [y_re1, y_re2, y_re3]

    

    def is_valid_date(self, candidate_date, date_format):
        '''判断是否是一个有效的日期字符串'''
        try:
            time.strptime(candidate_date, date_format)
            return True
        except:
            return False
        
    def convert_to_digital(self, raw_string):
        digital_string = ""
        for word in raw_string:
            if word in self.zh2num_dict.keys():
                digital_string += self.zh2num_dict[word]
            else:
                digital_string += word
        
        if digital_string.find("十")== -1:
            return digital_string
        
        digital_ten_1 = re.compile(u'([\d])十([\d])')
        digital_ten_2 = re.compile(u'([\d])十[^\d]*')
        digital_ten_3 = re.compile(u'十([\d])')
        
        while digital_string.find("十") != -1:
            if digital_ten_1.search(digital_string):
                match_obj = re.search(digital_ten_1, digital_string)
                aa1 = match_obj.group(1)
                aa2 = match_obj.group(2)
                new_result = str(int(aa1) * 10 + int(aa2))
                digital_string = digital_string.replace(aa1+"十"+aa2, new_result)
            elif digital_ten_2.search(digital_string):
                match_obj = re.search(digital_ten_2, digital_string)
                aa1 = match_obj.group(1)
                new_result = str(int(aa1) * 10)
                digital_string = digital_string.replace(aa1+"十", new_result)
                
            elif digital_ten_3.search(digital_string):
                match_obj = re.search(digital_ten_3, digital_string)
                aa1 = match_obj.group(1)
                new_result = str(10 + int(aa1))
                digital_string = digital_string.replace("十"+aa1, new_result)
    
            else:
                digital_string = digital_string.replace("十", "10")
                return digital_string
            
        return digital_string
        
        
      
    def run(self, raw_string):
        process_string = ""
        for word in raw_string:
            if word in self.zh2num_dict.keys():
                process_string += self.zh2num_dict[word]
            else:
                process_string += word
        
        process_string = self.convert_to_digital(raw_string)
        #抽取年、月、日信息
        for ymd_re in self.ymd_relist:
            if ymd_re.search(process_string):
                logging.info("日期属性值%s中包含年、月、日信息" %process_string)
                match_obj = re.search(ymd_re, process_string)
                year = match_obj.group(1)
                month = match_obj.group(2)
                day = match_obj.group(3)
                
                #日期补全 0000-00-00
                if len(year) != 4:
                    year = '0'*(4-len(year)) + year
                if len(month)==1:
                    month='0'+month
                if len(day)==1:
                    day='0'+day
                
                candidate_date = year + "年" + month + "月" + day + "日"
                
                if self.is_valid_date(candidate_date, "%Y年%m月%d日"):
                    return candidate_date
                else:
                    logging.info("%s不是一个有效的日期" %process_string)
                
        
        #抽取年、月信息
        for ym_re in self.ym_relist:
            if ym_re.search(process_string):
                logging.info("日期属性值%s中包含年、月信息" %process_string)
                match_obj = re.search(ym_re, process_string)
                year = match_obj.group(1)
                month = match_obj.group(2)
                
                #日期补全 0000-00
                if len(year) != 4:
                    year = '0'*(4-len(year)) + year
                if len(month)==1:
                    month='0'+month

                candidate_date = year + "年" + month + "月"
                
                if self.is_valid_date(candidate_date, "%Y年%m月"):
                    return candidate_date
                else:
                    logging.info("%s不是一个有效的日期" %process_string)
                
        
        #抽取月、日信息
        for md_re in self.md_relist:
            if md_re.search(process_string):
                logging.info("日期属性值%s中包含月、日信息" %process_string)
                match_obj = re.search(md_re, process_string)
                month = match_obj.group(1)
                day = match_obj.group(2)
                
                #日期补全 00-00
                if len(month)==1:
                    month='0'+month
                if len(day)==1:
                    day='0'+day
                
                candidate_date = month + "月" + day + "日"
                
                if self.is_valid_date(candidate_date, "%m月%d日"):
                    return candidate_date
                else:
                    logging.info("%s不是一个有效的日期" %process_string)
                
        
        #抽取年信息
        for y_re in self.y_relist:
            if y_re.search(process_string):
                logging.info("日期属性值%s中包含年信息" %process_string)
                match_obj = re.search(y_re, process_string)
                year = match_obj.group(1)
                
                #日期补全 0000-00-00
                if len(year) != 4:
                    year = '0'*(4-len(year)) + year
                
                candidate_date = year + "年"
                
                if self.is_valid_date(candidate_date, "%Y年"):
                    if process_string.find("公元前") != -1:
                        return "公元前" + candidate_date
                    return candidate_date
        
        return None