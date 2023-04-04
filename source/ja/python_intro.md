---
jupytext:
  formats: md:myst,ipynb
  notebook_metadata_filter: all
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.10.6
---

# 予備知識：Python

```{contents} 目次
---
local: true
---
```

+++

## 基礎編

Pythonに関する教材は世の中にいくらでもあるので、このワークブックで登場する概念・構文・テクニックに絞って説明をします。

```{admonition} すでにプログラミング経験がある場合
「CやJavaなどのプログラミング経験はあるが、Pythonは初めて」という人の理解を助けるために、随所でこのように注釈を入れます。プログラミング経験が一切ないという場合は無視して構いません。
```

### Pythonプログラムとは

プログラムとは、機械が順番に実行するべき命令を、決まった文法に従って書き下した文書のことです。プログラムには様々な言語があり、それぞれ用途や扱える命令群が異なります。Pythonは世界で最も広く使われている言語の一つで、扱える命令の多様さ、表現性の高さ、フレキシビリティが最大の特徴です。

### 行構造

最も単純には、Pythonのプログラムは一行一つの命令文で成り立ちます。各行で`#`が登場したら、それ以降のテキストは「コメント」として無視されます。また、空白（スペースやタブ）のみを含む行も無視されます。

```{code-cell} ipython3
:tags: [remove-output]

# aという変数に3という値を代入する命令文
a = 3
```

```{code-cell} ipython3
:tags: [remove-output]

# aに代入されている値と5を掛け算してbという変数に代入する命令文
b = a * 5
```

```{code-cell} ipython3
:tags: [remove-output]

# bの文字列表現をプリントアウトする命令文
print(b)
```

あとで解説する関数やクラスの定義、`if`、`for`、`try`/`except`などの複合文では、全ての行を一定数の空白文字分（通常スペース4つ、流儀によっては2つ。タブ文字は使うべきでない）だけインデントさせます。

```{code-cell} ipython3
:tags: [remove-output]

if b == 15:
    # bの値が15の時、インデントされている部分が実行される
    print('b is fifteen')
    print('This line is also executed')

else:
    # それ以外の場合、インデントされている部分が実行される
    print('b is some other number')
    print('This line is also not executed')

# インデンテーションが解消された＝if文を抜けた
print('This line is always executed')
```

インデントしたブロックの中で、さらにインデントを必要とする文を書く場合は、スペース8つ、12個、と入れ子状にインデントします。各レベルでのインデンテーション解消がブロックの終わりを意味します。

```{code-cell} ipython3
:tags: [remove-output]

if b == 2:
    # bの値が2の時、インデントされている部分が実行される

    for x in range(3):
        # さらにインデントされた部分が3回実行される
        print('b is 2 and I print this line three times')

    # 3回実行のループは抜けたが、if b == 2:のブロックの中
    print('b is 2 and I printed three times')

else:
    # それ以外の場合、インデントされている部分が実行される
    if a == 3:
        # そのうち、aの値が3の時はこの部分が実行される
        print('b is not 2 and a is 3')

# 全てのインデンテーションが解消された
print('All indentations resolved')
```

```{admonition} インデンテーション
C/C++やJavaなどではコードのブロックは{}括弧で括られるため、インデンテーションは単に人間がコードを読みやすくするための飾りです。Pythonではインデンテーションがプログラム上の意味を持ちます。
```

+++

### 変数

上で出てきた`a`や`b`を変数と呼びます。変数は（変「数」と書きますが）数値だけではなく、プログラム中で扱う様々なデータ（あとで言うオブジェクト）を表すラベルのようなものです。

```{admonition} 変数の生成
CやJavaなどの言語と異なり、Pythonでは変数を明示的に宣言しません。上のコードのように、値の代入をもって変数の宣言と初期化が行われます。また、変数は型を限定しないので、同じ変数名を別の型の値に使うことができます。
```

```{code-cell} ipython3
:tags: [remove-output]

# sという変数に「abc」という文字列を代入
# ' (single quote)もしくは " (double quote) で挟まれたテキストは文字列となる
s = 'Hello World'

print(s)
```

```{code-cell} ipython3
:tags: [remove-output]

# bという変数を真理値型の「真」で再定義
b = True
```

```{admonition} 文字列
C/C++と異なり、Pythonでは文字（character）と文字列（string）の区別がありません。文字情報は全て文字列です。また、' (single quote)と " (double quote)の間に機能的な違いもありません。
```

変数を使わないと、プログラムはただ「3」や「'Hello World'」のような「リテラル」（値がそのままプログラム文中に書き下されているデータ）で成り立つことになり、「3と5を足す」「Hello Worldと表示する」などの自明な動作しかできません。逆に言えば、変数を駆使することで初めてプログラムを書く意味が出てきます。

変数名はアルファベット一文字とは限りません。小文字のaからz、大文字のAからZ、アンダースコア（_）、0-9の数字の任意の長さの組み合わせが使えます。ただし、数字は変数名の最初の一文字には使えません。変数に意味のある名前をつけてコードを読みやすくすることは、プログラミングにおける重要なエチケットです。

```{code-cell} ipython3
:tags: [remove-output]

# 変数にはプログラム中でその変数が持つ役割をベースに名前をつける
dividend = 15
divisor = 3
quotient = dividend / divisor
```

### 演算子

数値に対して使える、様々な演算子が用意されています。

```{code-cell} ipython3
:tags: [remove-output]

# 足し算
print('a + 2 =', a + 2) # print()は中身を , (comma) で区切ることで、複数の表現をプリントアウトできる
# 引き算
print('a - 2 =', a - 2)
# 符号反転
print('-a =', -a)
# 括弧
print('-(-a) =', -(-a))
# 掛け算
print('a * 3 =', a * 3)
# 割り算
print('a / 2 =', a / 2)
# 切り捨て除算（整数の割り算における商）
print('a // 2 =', a // 2)
# 剰余
print('a % 2 =', a % 2)
# べき乗
print('a ** 3 =', a ** 3)
# 比較
print('a < 4 is', a < 4) # 4より小さい
print('a <= 3 is', a <= 3) # 3以下
print('a > 5 is', a > 5) # 5より大きい
print('a >= 1 is', a >= 1) # 1以上
print('a == 3 is', a == 3) # 3と等しい
print('a != 7 is', a != 7) # 7と等しくない
# 値の更新
a += 3 # aに3を足す
print('a += 3 -> a is', a)
a -= 4 # aから4を引く
print('a -= 4 -> a is', a)
a *= 5 # aに5をかける
print('a *= 5 -> a is', a)
a //= 2 # aを2で割る（切り捨て）
print('a //= 2 -> a is', a)
a /= 1.4 # aを1.4で割る
print('a /= 1.4 -> a is', a)
a %= 3 # aを3で割った剰余にする
print('a %= 3 -> a is', a)
a **= 3 # aを3乗する
print('a **= 3 -> a is', a)
```

他にもビットシフト演算子などが存在しますが、このワークブックでは使用しないので割愛します。

```{admonition} 割り算
Python 3では、/演算子は、整数同士に用いられたとしても、true divisionつまり浮動小数点数同士の割り算を引き起こします。
```

実は上の演算子の多くが数値以外にも使え、なんとなくそうなるかな、という直感的な効果を持ちます。Pythonの表現性が高い一つの理由がこれです。例えば文字列に対しては以下のようになります。

```{code-cell} ipython3
:tags: [remove-output]

# 文字列と文字列を足し合わせる -> 文字列の結合
print('"abc" + "xyz" = ', "abc" + "xyz")
# 文字列を整数倍する -> 文字列の反復
print('"ha" * 3 =', "ha" * 3)
# 文字列の比較 -> 辞書式順序での比較
print('"abc" < "def" is', "abc" < "def")
print('"xy" <= "xyz" is', "xy" <= "xyz")
# etc.
```

### 基本のデータ型

変数が表すデータには様々な「型」（タイプ）があります。文字と数は異なる、というのはプログラムに限らず現実世界でも成り立つ関係です。型ごとに行える操作が異なるので、データの型を常に意識しながらプログラムを書く必要があります。後で一般のクラスについて解説をするので、ここではPythonの組み込み型と言われる、言語レベルで定義されている基本的な型を紹介します。

#### 数値型 int, float, complex

数値を表す型には`int`（整数）、`float`（実数）、`complex`（複素数）があります。数学的には包含関係にある3つの型ですが、コンピュータ的には実数と複素数は「浮動小数点数」で実装されるため整数と根本的に異なります。複素数は浮動小数点数を2つ一組にし、複素数特有の演算規則を当てはめたものです。

3つの型は異なりますが、全て数値を表すものであるため、演算の互換性があります。型が混在する演算の結果は数学的な包含関係に従います。

```{code-cell} ipython3
:tags: [remove-output]

# 整数
i = 43
# 実数
f = 5.4
# 複素数
c = 2.3 + 0.9j

# 数学的には整数でも、小数点が入ると実数型になる
print('type of 7.0 is', type(7.0)) # type()はデータの型を返す関数

# intとfloatとの間の演算の結果はfloat
print('type of i + 8.0 is', type(i + 8.0))
# int/floatとcomplexとの間の演算の結果はcomplex
print('type of i + 2.0j is', type(i + 2.0j))
print('type of f * c is', type(f * c))
```

#### 文字列型 str

すでに上で登場していますが、' (single quote)もしくは " (double quote)で囲まれたテキストは文字列データとなります。二つの記号の間に機能上の違いはありません。なぜ二通り記号が用意されているかというと、プログラマ個人の嗜好に合わせられるという以外に、以下のようなコードを書きやすくするためです。

```{code-cell} ipython3
:tags: [remove-output]

print('The world says "Hello" in single quotes')
```

同じことを例えばdouble quotesだけで表現することもできます。その場合、文字列中のクオーテーション記号にバックスラッシュを付け、文字列を囲む記号と区別します。

```{code-cell} ipython3
:tags: [remove-output]

print("The world says \"Hello\" in double quotes")
```

上に比べて少しコードが読みにくくなることがわかります。

単純なquotesで囲まれた文字列はPythonプログラム中、一行で書ききらなければいけないので、書きたいテキストが改行を含む場合は改行文字`\n`（backslash-n）を挿入します。

```{code-cell} ipython3
:tags: [remove-output]

print('This text has\ntwo lines')
```

これもまた読みにくいので、実はsingle quotesとdouble quotes以外にそれらの記号を3つ並べた`'''`や`"""`（三重クオート）でテキストを囲み、改行を含む文字列をそのまま書くこともできます。

```{code-cell} ipython3
:tags: [remove-output]

s = '''This
is
   such
a
  long
string'''
print(s)
```

改行やクオーテーション以外にも「特殊文字」があり、全てバックスラッシュで始まります。実際、通常の文字列中でバックスラッシュが一つ登場すると、Pythonはそのバックスラッシュと次の文字（場合によっては複数字）を組み合わせて特殊文字として解釈します。なので、実際のバックスラッシュをテキスト中に登場させたい場合（LaTeXのテキストを書いているときなど）は、一つだけ書くと解釈されないので、`\\`という特殊文字を利用します。

```{code-cell} ipython3
:tags: [remove-output]

print('\\sqrt{2}')
```

これも読みにくいテキストになってしまうので、バックスラッシュがたくさん使われる文字列を書く場合は、`r`というプレフィックスをクオーテーションの前にくっつけます。`r`のついた文字列中では、バックスラッシュが特殊文字として解釈されません。

```{code-cell} ipython3
:tags: [remove-output]

print(r'\frac{e^{2 \pi i j k}}{\sqrt{2}}')
```

ときどき、文字列型以外の変数を文字列に変換する必要が出てきます。例えば、`"The value of variable x is "`という文字列の後に、`x`という変数の値を付け加えたいとします。`x`が整数型だと`+`演算子を使ったらエラーになります。

```{code-cell} ipython3
:tags: [remove-output, raises-exception]

s = 'The value of variable x is '
x = 3

# 文字列と整数を加えようとするとエラーになる
print(s + x)
```

文字列と整数の足し算は定義されていないためです。このような場合は、`str(x)`として`x`の10進数表現の文字列を作ります。

```{code-cell} ipython3
:tags: [remove-output]

# これは文字列同士の足し算
print(s + str(x))
```

ただ、この方法で変数の文字列表現をたくさん作ると、下のようにコードが冗長になりやすい問題があります。

```{code-cell} ipython3
:tags: [remove-output]

print('a is ' + str(a) + ' and b is ' + str(b) + ' and x is ' + str(x) + ' and so on')
```

そこで、Pythonでは文字列中に変数のプレースホルダーを入れ、後から変数値を代入することができるようになっています。ただ、歴史的経緯から、4つの異なる方法が実装されていて、少し厄介です。このワークブックでは2通りのみ使用することにします。

まず、原則的にはフォーマット済み文字列（f-string）と呼ばれる、歴史的には一番新しい方式を使います。f-stringを作るには、クオーテーションの前に`f`というプレフィックスをつけます。そして、文字列中の変数の値を代入したい部分に`{変数名}`を挿入します。

```{code-cell} ipython3
:tags: [remove-output]

print(f'a is {a} and b is {b} and x is {x} and so on')
```

f-stringは、実は変数に限らず、その場で評価可能な任意のプログラムコードを扱えます。中括弧の中にさらに文字列リテラルが入る場合は、全体の文字列に使っていないクオーテーション記号を使います。

```{code-cell} ipython3
:tags: [remove-output]

print(f'If I multiply a and x I get {a * x} and this is a {str(b) + " story"}')

# fとrを組み合わせて、バックスラッシュを含む文字列に変数値を代入することもできる
print(fr'\int_0^{a} x dx = {0.5 * a ** 2}')
```

このようにf-stringで大体の目的は達成できますが、厄介になるのは実際に中括弧を含む文字列を作りたい場合です。f-string中の中括弧は二重にして表現するので、簡単な文字列でもコードが非常に読みにくくなります。

```{code-cell} ipython3
:tags: [remove-output]

print(f'{{ brace }} {x}')

# 中括弧がたくさんあるととても読みにくい
print(fr'\frac{{e^{{{a}i}}}}{{2}}')
```

そこで、このワークブックでは中括弧を含む文字列に変数値を代入するときに限り、最も古い方式である%-formattingを使います。この方式でのプレースホルダーは、整数値を代入したいときは`%d`、実数値なら`%f`、文字列なら`%s`など、`%`文字と代入したいデータの型の識別子の組み合わせでできます。変数値を代入するには、文字列のすぐ後に`　%　変数`をくっつけます。

```{admonition} %-formatting
%を使った書式はC系やJavaのprintfと共通です。
```

```{code-cell} ipython3
:tags: [remove-output]

print(r'\frac{e^{%fi}}{2}' % a)
```

#### 配列型 tuple, list

複数のデータをひとまとめに並べておくと便利なことが多々あります。Pythonに最初から実装されている基本型のうち、重複を許して、順序つきでデータを集めた構造として、tuple（タプル）とlist（リスト）があります。Tupleは一度作ると要素を足したり除いたりできないのに対し、listは長さが可変で要素の入れ替えが可能です。

```{code-cell} ipython3
:tags: [remove-output]

# 空のtupleやlistを新しく作る
my_empty_tuple = tuple()
my_empty_list = list()

# データの入ったtupleやlistを新しく作る
# データの型が揃っている必要はない
my_tuple = (1, 2, 3, 'dah')
my_list = [2.3, 5. + 1.j, 2]
# コンストラクタ関数にリストやタプルを入れる
my_list = list(my_tuple)
my_tuple = tuple(my_list)
# 「ジェネレータ表現」を使う
my_tuple = tuple(x for x in ['a', 'b', 'c'])
my_list = list(x for x in range(4))
```

タプルやリストの要素を一つ参照するときは、鉤括弧でインデックス（要素の番号）を指定します。インデックスは0から始まるので、長さnの配列に対して有効なインデックスは0からn-1までです。

```{code-cell} ipython3
:tags: [remove-output]

t = ('one', 2, 'three')
print(t[0])
print(t[1])
print(t[2])
```

リストでは要素を新しい値にすることができます。

```{code-cell} ipython3
:tags: [remove-output]

print(my_list) # 数値や文字列以外もprint()できる
my_list[3] = 'three'
print(my_list)
```

あとで説明するfor文を使って、配列の要素を順番に参照することもできます。

```{code-cell} ipython3
:tags: [remove-output]

for element in my_list:
    print('Next element is', element)
```

タプルやリストの長さ（要素数）は`len()`関数でわかります。

```{code-cell} ipython3
:tags: [remove-output]

print(f'my_tuple has {len(my_tuple)} elements.')
print(f'my_list has {len(my_list)} elements.')
```

タプル同士やリスト同士は足し算で繋ぎ合わせられます。

```{code-cell} ipython3
:tags: [remove-output]

list1 = [0, 2]
list2 = ['hello'] # 長さ1のリスト
list3 = list1 + list2
print(list3)

tuple1 = (0, 2)
tuple2 = ('hello',) # 長さ1のタプル。最後のコンマがないと「文字列を括弧で括ったもの」になってしまうので注意
tuple3 = tuple1 + tuple2
print(tuple3)
```

数値に対してと同様、足し算での更新も可能です。

```{code-cell} ipython3
:tags: [remove-output]

list1 += list2
tuple1 += tuple2
print(list1)
print(tuple1)
```

リストの最後に新たな要素を足すには、`append()`というメソッド（後の解説を参照）を使います。

```{code-cell} ipython3
:tags: [remove-output]

list1.append('world')
print(list1)
```

すでに説明した通り、タプルには要素を足せないので、対応するメソッドがありません。

+++

#### 辞書型 dict

配列はリストが順番に入っている入れ物で、インデックスで要素を参照可能でした。辞書型は要素を「キー」（索引）で参照する入れ物です。キーとなりうるデータは数値、文字列、タプルなどです。

```{code-cell} ipython3
:tags: [remove-output]

# 空のdict
my_empty_dict = dict()

# データの入ったdict
# キーも要素もデータ型が揃っている必要はない
my_dict = {'a': 'AAAA', 3: 3333, tuple1: 'world'}
```

辞書型の要素を参照するときは鉤括弧にキーを入れます。辞書型と同様、要素の値を入れ替えることができます。また、全く新しいキーを加えることもできます。

```{code-cell} ipython3
:tags: [remove-output]

print('my_dict["a"] =', my_dict['a'])
my_dict[3] = 4444
my_dict[4] = 3333
print(my_dict)
```

辞書型に対しても`len()`が使えます。

```{code-cell} ipython3
:tags: [remove-output]

print(f'my_dict has {len(my_dict)} elements.')
```

#### 真理値型 bool

真理値を表す`bool`型には`True`または`False`の二通りの値しかありません。その代わり、様々な表現が真理値に明示的・暗示的に変換可能で、後述する`if`文などで利用されます。表現を明示的に真理値に変えるには`bool()`関数に表現を渡します。

```{code-cell} ipython3
:tags: [remove-output]

# 数値を真理値に変換。0、0.0、0.0+0.0j以外は全てTrue
print('bool(1) =', bool(1))
print('bool(0) =', bool(0))
print('bool(2.3) =', bool(2.3))
print('bool(0.) =', bool(0.)) # 小数点以下がゼロであるときにデータが実数型であることを表現するためには、小数点を末尾につける
print('bool(3.7+0.j) =', bool(3.7+0.j))
print('bool(0.+0.j) =', bool(0.+0.j))

# 文字列を真理値に変換。長さ0の文字列以外は全てTrue
print('bool("") =', bool(""))
print('bool("Hello") =', bool("Hello"))

# tuple, list, dictを真理値に変換。長さ（要素数）0のときFalse、それ以外はTrue
print('bool(my_tuple) =', bool(my_tuple))
print('bool(my_empty_tuple) =', bool(my_empty_tuple))
print('bool(my_list) =', bool(my_list))
print('bool(my_empty_list) =', bool(my_empty_list))
print('bool(my_dict) =', bool(my_dict))
print('bool(my_empty_dict) =', bool(my_empty_dict))
```

#### None型

Pythonプログラム中で特別な型・値として`None`があります。これはデータがない状態を表します。数値の`0`、文字列の`''`、真偽値の`False`などはそれぞれ型があり有効なデータなので、それらとは異なる「無」の状態を表すために`None`を使います。

+++

### 関数

プログラムは最悪全ての命令を一行ずつ書き下していけば走りますが、繰り返し行う動作や、論理上ひとまとまりにした方が全体の可読性が高まるコードの続きなどは、関数として定義します。すでに登場した`print()`や`len()`も関数です。

関数は引数（ひきすう、インプット）をとり、その値に応じて挙動を変えることができます。関数には返り値があり、関数を実行した結果として変数に代入したりできます。

```{admonition} 引数の型
Pythonでは変数が型に縛られないので、関数も引数の型を決めずに定義されます。どんな引数を渡しても、関数の内部のコードが引数のデータ型に対して有効でさえあれば、有効な関数となります。C++やJavaでいう関数のオーバーロードが自動で行われているとも考えられます。
```

関数の定義には`def`というキーワードを使い、コードを一段インデントして記述します。関数名に使える文字は変数名のものと同じです。

```{code-cell} ipython3
:tags: [remove-output]

# myfunctionという名の関数を定義する。argは引数で、「関数を呼んだときに渡される最初の引数の値が関数内のコードでargという変数として使用される」ことを意味する
def my_function(arg):
    # 渡された引数をprintする
    print(arg)
    # 引数に2を足した値を返す
    return arg + 2

# myfunctionをaを引数として実行し、返ってきた値をa_plus_2に代入する
a_plus_2 = my_function(a)
print('Returned value:', a_plus_2)
```

関数の引数にはデフォルトの値を持たせることができます。そのような引数には、関数実行時に値を指定されなければ、設定されたデフォルト値が代入されます。

また、関数は、関数が定義されている部分のコード中で定義されている変数を参照することができます。

```{admonition} 変数のスコープ
Pythonの変数のスコープの考え方はC/C++などと異なるので注意が必要です。
```

```{code-cell} ipython3
:tags: [remove-output]

# arg2にはデフォルトで4という値が入る
def another_function(arg1, arg2=4):
    return arg1 + arg2

# arg2の値を指定しなかったので、4が使われる
print(another_function(3))
# arg2を指定するとその値が使われる
print(another_function(1, 5))

# 関数の外で定義された変数
outer_scope_variable = 'Q'

# 引数を取らない関数もある
def yet_another_function():
    # 関数の中から外の変数を参照可能
    return outer_scope_variable + 'uantum'

print(yet_another_function())
```

### if文（条件文）

プログラム中で、変数の値に応じて実行する内容を変えたい時はif文で条件分岐をすることができます。分岐したプログラムのコードは一段インデントして記述します。分岐したコードは独立なので、別々の変数を定義したり、全く異なる関数を実行したりできます。ただし一方でしか定義されない変数を作る場合は、もう一方のケースでその変数が参照されないか注意が必要です。

```{code-cell} ipython3
:tags: [remove-output]

# ifの後は一つ以上スペースを空けて、真偽値となる表現が入る
if len(my_dict) == 4:
    # my_dictの長さが4である場合
    true_case = my_function(7)
else:
    # my_dictの長さが4でない場合
    false_case = another_function(1, 3)
```

```{code-cell} ipython3
:tags: [remove-output]

# 有効
print(f'len(my_dict) was {len(my_dict)} so if I try to reference true_case I get this: {true_case}')
```

```{code-cell} ipython3
:tags: [raises-exception, remove-output]

# エラーを起こす
print(f'len(my_dict) was {len(my_dict)} so if I try to reference false_case I get this: {false_case}')
```

```{admonition} 変数のスコープ
if文は独立のスコープを持たないので、上のようなことが起こります。下で登場するforやwhileでも同様です。
```

上でさまざまな型を`bool()`で真偽値に変換しましたが、if文の条件部分ではこの変換が自動で行われます。

```{code-cell} ipython3
:tags: [remove-output]

print('Conditional expression on the length of my_empty_tuple:')

if my_empty_tuple:
    # my_empty_tupleの長さが0でないケース
    print('my_empty_tuple is actually not empty')

# 条件文にはelseがなくてもいい
```

### for文、while文（ループ）

同じコードを（条件を変えながら）何度も実行したいときは、for文やwhile文を使ったループを作ります。

```{code-cell} ipython3
:tags: [remove-output]

# リストの要素についてループ。インデントされている部分が要素数だけ繰り返し実行され、変数elementに各要素が代入される
for element in my_list:
    print(f'another_function({element}, {element}) = {another_function(element, element)}')

# 0から5までの整数についてループ
for index in range(6):
    print(f'another_function({index}) = {another_function(index)}')

# 配列の各要素が一定の長さの配列である場合、要素配列の要素を直接ループ変数に代入できる
list_of_tuples = [('x', 10), ('y', 6)]
for char_var, num_var in list_of_tuples:
    print(f'char_var = {char_var}, num_var = {num_var}')

# my_stringの長さが10以上になるまでループ
my_string = ''
while len(my_string) < 10: # 文字列の長さもlen()でわかる
    my_string += 'abc'

print(my_string)
```

### クラス、オブジェクト、メソッド

上で基本のデータ型を紹介しましたが、Pythonではプログラム中で新たに型を定義することができます。型のことをクラスともいい、これまでデータと呼んできた構造をオブジェクトともいいます。全てのオブジェクトは何らかのクラスの一つの具象化（インスタンス）です。

このワークブックでは新たにクラスを定義することはありませんが、参考のためにクラスの記述の仕方を紹介します。

```{code-cell} ipython3
:tags: [remove-output]

class EmptyClass:
    pass # passは「文法上インデントしなければいけないけどプログラムとして何もしない」ことを表す
```

このままでは`EmptyClass`というクラスがプログラムの中で定義されているという他、何の役割も果たしません。プログラムによってはあるクラスが存在するだけでいいということもありますが、通常は属性（インスタンスに付随するデータ）を足したり、「メソッド」を定義したりします。メソッドというのはクラスに対して定義される関数のことです。インスタンスを自動的に引数に取るため、挙動がインスタンスごとに異なります。

```{code-cell} ipython3
:tags: [remove-output]

class MoreFunctionalClass:
    # __init__は特殊メソッド（コンストラクタ）で、インスタンス生成時に呼ばれる
    def __init__(self, x, y):
        # メソッドの最初の引数selfはインスタンスを指す
        self.attr_x = x
        self._y = y

    # メソッドの一例
    def add_y_to_x(self):
        return self.attr_x + self._y

# MoreFunctionalClassのインスタンスを作り、変数mに代入する
m = MoreFunctionalClass(2, 5)

# インスタンスの属性を参照する
print(f'Value of attr_x of m is {m.attr_x}')

# インスタンスのメソッドを実行する
# 引数selfは自動で渡される
print(f'Calling add_y_to_x on m: {m.add_y_to_x()}')
```

### モジュールのインポート

通常、Pythonプログラムは一つのファイルに書ききられません。コードが長くなりすぎるだけでなく、定義される関数やクラスが他のプログラムでも利用できる場合があるからです。別ファイルで定義された変数、関数、クラスなどをプログラム中で使用するには、`import`という命令を使い、ファイルの名前や、場合によってはファイル中の使用したい特定の関数名などを指定します。`import`で取り込まれるファイルをPythonモジュールと呼びます。

```{code-cell} ipython3
:tags: [remove-output]

# osというスタンダードライブラリのモジュールをインポート
import os
# osモジュールで定義されているgetcwd()という関数（現在どのディレクトリからプログラムを実行しているかを返す）を呼ぶ
print(os.getcwd())

# qiskit（IBMの量子コンピューティング用ライブラリ）からQuantumCircuitクラスをインポート
from qiskit import QuantumCircuit

# numpy（数値計算用ライブラリ）をインポートし、プログラム中npという短縮したモジュール名で使用
import numpy as np
```

Pythonの非常に大きな強みが、ユーザーベースの広さとオープンソースのライブラリの多さです。Pythonでやりたいと思うことは大体誰かがすでに実装していて、[PyPI](https://pypi.org)などでパッケージとして公開しているので、ユーザーはパッケージをインストールしてプログラム中からモジュールを`import`すればいい、という具合です。

+++

### 例外（exception）

プログラムで想定していない操作が要求されたときには「例外」が発生します。上の例では、文字列と整数を加えようとしたときや、if文の分岐によって定義されていない変数を参照しようとしたときに例外が発生しました。例外が発生するとプログラムの実行が中断されますが、これは実は必ずしも「クラッシュ」ということではありません。Pythonには例外の発生を検知して対処できる仕組みが備わっています。例外処理をするには、例外が発生する可能性のあるコードを`try:`というキーワードで始まるブロックに入れ、直後に`except`というキーワードから始まるブロックに対処のためのコードを記述しておきます。

```{code-cell} ipython3
:tags: [remove-output]

s = 'The value of variable x is '
x = 3

# try以下のブロックで例外が発生すると、ブロック中のコードの実行がその時点で止まり、後ろのexceptブロックが実行される
try:
    # 文字列と整数を加えようとするとエラーになる
    print(s + x)

except TypeError: # TypeError型の例外が発生したら、以下のブロックを実行する
    print(f'Oops, {type(s)} and {type(x)} are not compatible')

except: # その他の型の例外が発生したら、以下のブロックを実行する
    print(f'This was unexpected: We got a {ex}')
```

上の例にあるように、例外にも型があり、例外発生のさまざまな理由を区別できるようになっています。例外とその処理はPythonの正常なプログラムフローの一部として捉えるべきで、`try``except`を上手に使うことでより「Pythonらしい」コードが書けると言われています。

例えば、辞書型に対して存在しないキーで値を参照しようとすると`KeyError`が発生します。これを回避するにはキーが存在するか事前に確認すればいいわけですが、確認した結果キーが存在したとして、その後値を参照するとすると、結局二回同じキーを検索することになります。それなら、最初から参照しようとしてみて、エラーが出たらキーが存在しなかったときの処理に移る、という方が実は効率的です。

```{code-cell} ipython3
:tags: [remove-output]

d = {'this': 3, 'is': 89, 'a': 123, 'dictionary': 98}

# dにキー'a'があるかどうか確認し、それからその値を使う。'a'がなければ他のことをする
# キーの存在を調べるにはinという演算子を使う
if 'a' in d:
    a_value = d['a']
else:
    print('d does not have an entry for "a"')
    a_value = None

# もっと効率的な実装
try:
    a_value = d['a']
except KeyError:
    print('d does not have an entry for "a"')
    a_value = None
```

## 実用編

Pythonの文法や基本概念を一通りさらったので、実際にワークブックに登場するようなプログラムを書くのに必要なものの紹介に移ります。

### 実習で登場する組み込み関数/型（Built-in functions）

組み込み関数とは、`import`なしでPython言語に最初から定義されている関数のことです。

```{code-cell} ipython3
:tags: [remove-output]

# abs: 数値の絶対値を返す
abs(-2.3) # 2.3
abs(1.2+1.6j) # 複素数の絶対値（sqrt(real * real + imag * imag)）-> 2.0

# enumerate: 配列のインデックスと要素を順番に返す
l = ['a', 'b', 'c']
for idx, element in enumerate(l):
    # ループの繰り返しごとに、idx（インデックス）は値 0, 1, 2 をとり、elementは 'a', 'b', 'c' になる
    pass

# isinstance: 変数やリテラルが特定のクラスのオブジェクトかチェックする
isinstance(l, list) # True
isinstance(2.3, int) # False

# len: 配列や辞書の長さを返す
len(l) # 3

# max: 引数の中から最大のものを返す
# min: 引数の中から最小のものを返す
max(3, 1, 9) # 9
min('b', 'x', 'c') # 比較演算子が定義されている型であれば成立 -> 'b'
max(l) # 引数が配列の場合、要素が比較される -> 'c'

# print: 文字列を表示する
print("We've already seen this")

# range: 一定間隔の整数の配列を作る
range(5) # 0, 1, 2, 3, 4
range(78, 92) # 78から91までの14個の整数の配列
range(3, 10, 2) # 3以上10未満、間隔2の整数の配列 = 3, 5, 7, 9
range(4, 1, -1) # 間隔は負でもよい。4, 3, 2

# sum: 配列の要素の総和を返す
sum([5, 6, 7]) # 18

# zip: 複数の配列から要素をタプルにまとめた配列を作る
m = [9, 4, 5]
for char, num in zip(l, m):
    # [(l[0], m[0]), (l[1], m[1]), ...]のforループと等価
    pass
```

文字列（`str`）、リスト（`list`）、タプル（`tuple`）、辞書（`dict`）などの組み込み型でよく使うメソッドも紹介しておきます。

```{code-cell} ipython3
:tags: [remove-output]

# str.join: 文字列の配列をつなげる
', '.join(['apples', 'bananas', 'bonobos']) # 'apples, bananas, bonobos'

# list.append: 配列の最後に要素を足す
[0, 1].append(2) # [0, 1, 2]

# list.insert: 配列の任意の位置に要素を挿入する。最初の引数が位置、次の引数が要素
['x', 'y'].insert(0, 'a') # ['a', 'x', 'y']

# dict.keys: 辞書のキーの配列を作る
d = {'this': 3, 'is': 89, 'a': 123, 'dictionary': 98}
for key in d.keys():
    # keyが'this', 'is', 'a', 'dictionary'のいずれかの値になる
    # ただし、キーがどの順番で現れるかは（特殊な場合を除いて）不確定
    pass

# dict.values: 辞書の値の配列を作る
for value in d.values():
    # valueが3, 89, 123, 98のいずれかの値になる
    # keysと同じく順序は不確定
    pass

# dict.items: 辞書のキーと値のペアの配列を作る
for key, value in d.items():
    # keysと同じく順序は不確定
    pass

# dict.get: [key]と同じように特定のキーに対応する値を返すが、キーが存在しない場合にデフォルトの返り値を設定できる
d.get('that', 39) # 39 (default value)
```

### NumPy

Pythonで数値計算をする際に今や欠かせない存在になっているのが、NumPy（「なむぱい」もしくは「なむぴー」）ライブラリです。NumPyは基本的には数値の（多次元）配列に対して効率的に数値計算をすることを目的に書かれていますが、サポートされている計算オペレーションの多様さも支持を広げる要因になっています。

このワークブックで利用するNumPyの機能について解説します。あまり複雑なものは使用していません。まずは`numpy`をインポートしますが、慣習的に`np`と短縮したモジュール名を使います。

```{code-cell} ipython3
:tags: [remove-output]

import numpy as np
```

#### 配列を作る

numpyの配列は`ndarray`という独自のクラスのオブジェクトで、リストやタプルとは異なります。

`ndarray`に収められているデータにも整数、実数、複素数といった型があります。整数にも8ビット、16ビット、32ビットとあるなど、Pythonで定義されているよりも型の種類が細かく定められていますが、普段使うのは64ビット整数、64ビット実数、128ビット複素数、ブーリアン（真偽値）、あたりです。

```{code-cell} ipython3
:tags: [remove-output]

# データから直接作る
arr = np.array([0., 1., 2.5, 7.]) # numpyの配列では要素のデータ型が揃っている必要がある。ただし整数と実数が混ざっていれば実数の、複素数が混ざっていれば複素数の配列が作られる

# 配列を作る関数から作る
arr = np.arange(4) # 組み込みrange(4)のように、[0, 1, 2, 3]という配列を作る
arr = np.linspace(0., 1., 100) # 長さ100の配列で、0.0から1.0まで等間隔に値を取る
arr = np.zeros(5, dtype=float) # 長さ5の配列で、全ての要素が0.0。dtype=floatはデータ型を陽に実数と決めるための引数
arr = np.ones(7, dtype=bool) # 長さ7の配列で、全ての要素が1。ただしdtype=boolとしたので、1の真偽値変換（True）が入る
arr = np.empty(4) # 長さ4の配列で、値が初期化されていない

# さまざまな配列をndarrayに変換する
m = [9, 4, 5]
arr = np.asarray(m)
arr2 = np.asarray(arr) # ndarrayをasarrayに入れても何も起こらない（arrとarr2は同じオブジェクト）

# ある配列と同じ形（次元数と各次元の要素数）の配列を作る
arr2 = np.zeros_like(arr)
arr2 = np.ones_like(arr)
arr2 = np.empty_like(arr)
```

#### 配列の要素を参照する

```{code-cell} ipython3
:tags: [remove-output]

# 要素数2x4の2次元配列を生成
arr = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
print('The full array:')
print(arr)

# 1次元目のインデックス0、2次元目のインデックス3の要素を選ぶ
print(f'arr[0, 3] = {arr[0, 3]}')

# 1次元目のインデックス1の要素（＝2次元目に相当する配列）
print(f'arr[1] = {arr[1]}')

# 1次元目のインデックス1、2次元目は0から2つおきに抽出
print(f'arr[1, 0:4:2] = {arr[1, 0:4:2]}')
```

#### 配列を操作する

```{code-cell} ipython3
:tags: [remove-output]

# reshape: 配列の次元数や各次元の要素数を変える
arr = np.arange(12) # 要素数12の1次元配列
print(arr)
rarr = arr.reshape(3, 4) # 要素数3x4の2次元配列。[0, 0]が0、[0, 1]が1、...
print(f'reshaped: {rarr}')

# concatenate: 配列をつなげる
brr = np.ones(4, dtype=int)
crr = np.concatenate((arr, brr))
print(f'concatenated: {crr}')
```

#### 配列に対して計算を施す

NumPyの大きな強みの一つが、さまざまな計算オペレーションが配列全体に対して同時に実行されることです。例えば`[0, 1, 2]`と`[3, 4, 5]`という配列があり、それらの要素同士の足し算をしたいとしたら、配列がPythonの`list`や`tuple`型であった場合のコードは

```{code-cell} ipython3
:tags: [remove-output]

list1 = [0, 1, 2]
list2 = [3, 4, 5]

list1 + list2 # これではただリストが繋がるだけなので

# ループを回して要素ごとに足し算をする必要がある
list3 = []
for i1, i2 in zip(list1, list2):
    list3.append(i1 + i2)

print(list3)
```

のようになります。これだと簡単なことをするために何行もコードがいるだけでなく、実は非常に非効率です。というのも、Pythonには、コード一行一行の実行が比較的遅いという弱点があるからです。これはPythonがCやJavaのようにプログラム全体を「コンパイル」して最初から機械言語に変換して実行する言語ではないことに起因します。上のようにループを回すと、ループの繰り返しごと、一行ごとに書かれている命令を解釈し直すので、時間がかかります。無論繰り返し数3のループではCなどとPythonの間で人間が感知できるような時間の差はありませんが、配列の大きさが例えば数十万を超えてくると、明らかに違いが見えてきます。

リストを使った上のケースに対して、NumPyの`ndarray`を使った場合は

```{code-cell} ipython3
:tags: [remove-output]

arr1 = np.array([0, 1, 2])
arr2 = np.array([3, 4, 5])

arr3 = arr1 + arr2 # これで要素ごとの足し算が行われる

print(arr3)
```

となります。コードの行数が少なくなっただけでなく、実はこの場合はCなどと同じ速さで計算が行われます。NumPyはPythonから扱えるモジュールですが、中身はコンパイルされて極度に最適化されたライブラリなのです。

配列に対して行える計算には要素ごとの四則演算だけでなく、さまざまな数学的関数もあります。

```{code-cell} ipython3
:tags: [remove-output]

# 要素ごとの引き算、掛け算、割り算
arr1 - arr2
arr1 * arr2
arr1 // arr2
arr1 / arr2

# 配列と一つの数（スカラー）との演算。配列の全ての要素に同じ演算が施される
arr1 + 2
arr1 - 3
arr1 * np.pi # np.pi = 3.141592653589793
arr1 // 2
arr1 / np.e # np.e = 2.718281828459045
arr1 ** 3

# 数学的関数の数々
np.exp(arr1)
np.log(arr2)
np.log2(arr2) # 2を底とする対数
np.cos(arr1)
np.sin(arr1)
np.tan(arr1)
np.arccos(np.linspace(0., 1., 20))
np.arcsin(np.linspace(0., 1., 20))
np.arctan(arr1)
np.square(arr1) # 全ての要素を2乗する
np.sqrt(arr1) # 全ての要素のルートを取る
np.ceil(arr1) # ceiling (値以上で最も小さい整数)
np.floor(arr1) # floor (値以下で最も大きい整数)
```

要素ごとではなく配列全体にかかる関数もあります。

```{code-cell} ipython3
:tags: [remove-output]

# 配列の全要素を足し上げる
np.sum(arr2)

# 配列の要素の平均値
np.mean(arr2)
```

NumPyで実装されている関数はあまりにも多いので、このワークブックで使われている分でもここで紹介しきれていないかもしれません。残りは[NumPyのドキュメンテーション](https://numpy.org/doc/stable/)を参照してください。

+++

### Matplotlib

NumPyで行った数値計算の結果を可視化するとき、おそらく最もよく用いられるのがMatplotlibというグラフ描画ライブラリです。このワークブックではMatplotlibのごく限られた機能しか使わないので、ここでは最低限の説明にとどめておきます。

Matplotlibのモジュール名は`matplotlib`ですが、また慣習的に`mpl`と略してインポートすることが多いようです。また、`matplotlib`そのものよりも、そのサブモジュールである`matplotlib.pyplot`の方をインポートすることが多く、こちらもまた慣習的に`plt`と略されます。

```{code-cell} ipython3
:tags: [remove-output]

%matplotlib inline
import matplotlib.pyplot as plt
```

Matplotlibには「図」（Figure）と「グラフ描画領域」（Axes）という概念があります。一つのFigureの中に複数のAxesが存在でき、各Axesに一つのプロットが描画される構造になっています。

このワークブックでは新しい図を作るのに二通りのコードが登場します。

```{code-cell} ipython3
:tags: [remove-output]

# 空っぽのFigureを作る
fig = plt.figure()
# Axesを一つ作る
ax = fig.add_subplot()

# Figureを作ると同時に、中身のAxesも作り、配置する
fig, axs = plt.subplots(2, 3) # 縦に2列、横に3列の合計6個のAxesが生成される。figはFigureオブジェクト、axsはAxesオブジェクトの配列
```

こうして作ったFigureやAxesオブジェクトの様々なメソッドを使ってグラフを描くことができますが、実は単にxとyのデータがあって一つスキャッタープロットを作りたいという時などは、もっと簡単な方法を取ることもできます。

```{code-cell} ipython3
:tags: [remove-output]

xdata = np.linspace(0., 3. * np.pi, 100)
ydata = np.cos(xdata)

# 折れ線グラフを描く
plt.plot(xdata, ydata)
```

```{code-cell} ipython3
:tags: [remove-output]

xdata = np.array([1., 3.3, 2.5])
ydata = np.array([9.6, 7.4, 5.5])

# スキャッタープロットを描く
plt.scatter(xdata, ydata, label='points')
plt.legend()
```

`plt.plot`や`plt.scatter`を複数実行すると、同じAxesにグラフが重なってプロットされます。実行の際に`label='some text'`という引数を足し、`plt.legend()`を呼ぶと、凡例が表示されます。また`plt.xlabel()`や`plt.ylabel()`でX, Y軸にタイトルをつけたり、`plt.title()`でプロット全体にタイトルをつけたりできます。

+++

### Jupyter

Pythonはもともとコマンドラインからスクリプト・プログラムとして実行したり、プロンプトを通じて一行ずつ動かしたりするようにできていますが、近年では（今このノートブックを表示・実行している）JupyterやVisual Studioなどの実行環境を通じてインタラクティブかつグラフィカルにプログラミングができるようになりました。

中でもJupyterはIPythonというライブラリと深く関係しており、ブラウザ上で快適にPythonを動かす様々なツールを提供しています。その中から、ここではこのワークブックで登場する数式や画像を表示させるための関数だけを紹介します。

```{code-cell} ipython3
# デモのためわざと図の自動描画機能をオフにする
%matplotlib
from IPython.display import display

# subplots()に何も引数を渡さないと、Axes一つのFigureができる
fig, ax = plt.subplots()

# スキャッタープロットを描く
xdata = np.array([1., 3.3, 2.5])
ydata = np.array([9.6, 7.4, 5.5])
ax.scatter(xdata, ydata, label='points')
ax.legend();
```

```{code-cell} ipython3
# 自動描画されない図はdisplay()で表示できる
display(fig)
```

```{code-cell} ipython3
# LaTeXで書いた数式をタイプセットする
from IPython.display import Math

Math(r'\frac{e^{2 \pi i j k}}{\sqrt{2}}')
```
