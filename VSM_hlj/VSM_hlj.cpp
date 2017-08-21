// VSM_hlj.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <algorithm>
#include <hash_map>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
const int MAX = 3333;
//字符串分割函数
std::vector<std::string> split(std::string str, std::string pattern)
{
	std::string::size_type pos;
	std::vector<std::string> result;
	str += pattern;//扩展字符串以方便操作
	int size = str.size();

	for (int i = 0; i<size; i++)
	{
		pos = str.find(pattern, i);
		if (pos<size)
		{
			std::string s = str.substr(i, pos - i);
			result.push_back(s);
			i = pos + pattern.size() - 1;
		}
	}
	return result;
}
int _tmain(int argc, _TCHAR* argv[])
{
	vector<hash_map<string, int >> words_num;//含词频的文档
	hash_map<string, int > term;//特征词
	vector<vector<string>> docs;//文档
	hash_map<string, int > stopWords;//停用词
	string line;
	vector<string> words;
	/*读取停用词文件*/
	int count = 1;
	ifstream in_;
	in_.open("Chinese-StopWords.txt");
	if (!in_)
	{
		cout << "Chinese-StopWords.txt open error!" << endl;
		return 0;
	}
	while (!in_.eof())
	{
		getline(in_, line);
		//cout << line << endl;
		stopWords.insert(pair<string, int>(line, count++));
	}
	in_.close();
	//cout << stopWords.size() << endl;
	/*读取文档文件*/
	int len = 0;
	count = 1;
	hash_map<string, int>::iterator s;
	hash_map<string, int>::iterator t;
	ifstream in;
	in.open("input.txt");
	if (!in)
	{
		cout << "input.txt open error!" << endl;
		return 0;
	}
	//int nnn = 0;
	while (!in.eof())
	{
		getline(in, line);
		words = split(line, " ");//用空格分词
		len = words.size();
		vector<string> doc;
		hash_map<string, int > word_num;//单个文档的词及词频
		for (int i = 1; i < len; i++)
		{
			//去掉停用词
			s = stopWords.find(words[i]);//查停用词
			if (s == stopWords.end())//不是停用词
			{
				doc.push_back(words[i]);
				//构建term
				t = term.find(words[i]);
				if (t == term.end())//不存在，将新的词添加到term中
				{
					term.insert(pair<string, int>(words[i], count));
				}
				else//存在，将该词的词频+1
				{
					term[words[i]] = term[words[i]] + 1;
				}
				//构建文档：词与词频
				t = word_num.find(words[i]);
				if (t == word_num.end())//不存在，将新的词添加到word_num中
				{
					word_num.insert(pair<string, int>(words[i], count));
				}
				else//存在，将该词的词频+1
				{
					word_num[words[i]] = word_num[words[i]] + 1;
				}
			}
		}

		docs.push_back(doc);
		words_num.push_back(word_num);
		doc.clear();
		word_num.clear();
	}
	cout << "文档总数为："<<docs.size() << endl;
	cout << "term总数为："<<term.size() << endl;
	//cout << nnn << endl;
	in.close();
	//cout << terms.size() << endl;
	/*term降维  --  选择出现次数较多的词作为term*/
	t = term.begin();
	while (t != term.end())
	{
		if (t->second < 2)//选择出现次数>=3的词作为term
		{
			s = t;
			t++;
			term.erase(s);
			continue;
		}
		t++;
	}
	cout <<"降维后，term总数为：" <<term.size() << endl;
	/*权重计算 TF IDF*/
	ofstream out_;
	out_.open("tf_idf.txt");
	//double matrix[MAX][MAX];//栈溢出
	vector<vector<double>> TF_IDF;
	vector<hash_map<string, int >>::iterator w;
	double tf, idf,n,sum,tf_idf,d,j;
	d = docs.size();//语料库中的文件总数
	for (int i = 0; i < d; i++)
	{
		vector<double> temp;
		for (t = term.begin(); t!=term.end(); t++)
		{
			//tf=n/sum
			n = words_num[i][t->first];//每个term出现在每个doc里的词频
			sum = docs[i].size();//每个doc共有多少词数
			tf = n / sum;
			//idf=log(d/j)
			j = 0;//
			for (w = words_num.begin(); w != words_num.end(); w++)
			{
				s = w->find(t->first);
				if (s != w->end())//存在
				{
					j++;
				}
			}
			idf = log(d / j);
			tf_idf = tf*idf;
			out_ << tf_idf;
			out_ << " ";
			temp.push_back(tf_idf);
		}
		out_ << "\n";
		TF_IDF.push_back(temp);
		temp.clear();
	}
	out_.close();
	cout << "相似度大于0.8的文档(除了自身与自身)统计如下：" << endl;
	//相似度计算
	//sim(d,p)=d*p/(|d|*|p|)
	ofstream out;
	out.open("outs.txt");
	if (!out)
	{
		cout << "open out.txt error!" << endl;
		return 0;
	}
	int  term_num = term.size();
	double dp,m2,n2,sim;
	int sim_num = 0;
	for (int i = 0; i < d; i++)
	{
		m2 = 0.0;
		for (int k = 0; k < term_num; k++)
		{
			m2 += TF_IDF[i][k] * TF_IDF[i][k];//|d|
		}
		for (int j = i+1; j < d; j++)
		{
			dp = 0;
			n2 = 0.0;
			for (int k = 0; k < term_num; k++)
			{
				dp += TF_IDF[i][k] * TF_IDF[j][k];//d*p
			}
			for (int k = 0; k < term_num; k++)
			{
				n2 += TF_IDF[j][k] * TF_IDF[j][k];//|p|
			}
			sim = dp / (sqrt(m2)*sqrt(n2));
			out << i;
			out << " ";
			out << j;
			out << ":";
			out << sim;
			out << "\n";
			if (sim > 0.8)
			{
				cout << "第 " << i << " 篇文档与第 " << j << " 篇的文档之间的相似度为：" << sim << endl;
			}
		}
	}
	cout << "相似度大于0.8的文档(除了自身与自身)总共：" <<sim_num<< endl;
	out.close();
	return 0;
}

