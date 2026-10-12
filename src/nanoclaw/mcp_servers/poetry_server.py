"""
古诗词查询 MCP Server

基于 MCP 官方 Python SDK 的服务端最小实现，用于演示/测试 NanoClaw 的
MCP 集成能力。运行方式：以 stdio 传输阻塞运行（python poetry_server.py），
由客户端（如 NanoClaw 的 MCPClientManager）按配置以子进程拉起，
通过 stdin/stdout 交换 JSON-RPC 消息。

FastMCP 服务端实现要点：
- 只需创建 ServerImpl 实例，再用 @mcp.tool() 装饰普通 Python 函数即可
  将其暴露为 MCP 工具，无需手写 JSON-RPC 处理逻辑；
- 框架根据函数的类型注解自动生成 inputSchema，从 docstring 自动提取
  工具描述与参数说明——因此每个工具函数都写了规范的 Args/Returns docstring；
- 本服务器提供古诗词搜索、随机推荐和诗人列表查询三个工具。
"""

import random

# 兼容 mcp 1.x / 2.x：2.x 中 FastMCP 更名为 MCPServer，导入路径也调整
try:
    from mcp.server.mcpserver import MCPServer as ServerImpl
except ImportError:
    from mcp.server.fastmcp import FastMCP as ServerImpl

# 实例化 MCP 服务器，名称 "poetry-server" 会在 initialize 握手时上报给客户端用于标识
mcp = ServerImpl("poetry-server")

# 古诗词数据源（唐诗宋词）：演示用内存数据，避免引入数据库等外部依赖，
# 十余首规模足以验证搜索/随机/去重三种工具的行为
POEMS = [
    {
        "title": "静夜思",
        "author": "李白",
        "dynasty": "唐",
        "content": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
    },
    {
        "title": "春晓",
        "author": "孟浩然",
        "dynasty": "唐",
        "content": "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
    },
    {
        "title": "登鹳雀楼",
        "author": "王之涣",
        "dynasty": "唐",
        "content": "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"
    },
    {
        "title": "相思",
        "author": "王维",
        "dynasty": "唐",
        "content": "红豆生南国，春来发几枝。愿君多采撷，此物最相思。"
    },
    {
        "title": "望庐山瀑布",
        "author": "李白",
        "dynasty": "唐",
        "content": "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。"
    },
    {
        "title": "江雪",
        "author": "柳宗元",
        "dynasty": "唐",
        "content": "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。"
    },
    {
        "title": "清明",
        "author": "杜牧",
        "dynasty": "唐",
        "content": "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。"
    },
    {
        "title": "水调歌头·明月几时有",
        "author": "苏轼",
        "dynasty": "宋",
        "content": "明月几时有？把酒问青天。不知天上宫阙，今夕是何年。我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆？人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。"
    },
    {
        "title": "念奴娇·赤壁怀古",
        "author": "苏轼",
        "dynasty": "宋",
        "content": "大江东去，浪淘尽，千古风流人物。故垒西边，人道是，三国周郎赤壁。乱石穿空，惊涛拍岸，卷起千堆雪。江山如画，一时多少豪杰。遥想公瑾当年，小乔初嫁了，雄姿英发。羽扇纶巾，谈笑间，樯橹灰飞烟灭。故国神游，多情应笑我，早生华发。人生如梦，一尊还酹江月。"
    },
    {
        "title": "声声慢·寻寻觅觅",
        "author": "李清照",
        "dynasty": "宋",
        "content": "寻寻觅觅，冷冷清清，凄凄惨惨戚戚。乍暖还寒时候，最难将息。三杯两盏淡酒，怎敌他、晚来风急？雁过也，正伤心，却是旧时相识。满地黄花堆积。憔悴损，如今有谁堪摘？守着窗儿，独自怎生得黑？梧桐更兼细雨，到黄昏、点点滴滴。这次第，怎一个愁字了得！"
    },
    {
        "title": "虞美人·春花秋月何时了",
        "author": "李煜",
        "dynasty": "宋",
        "content": "春花秋月何时了？往事知多少。小楼昨夜又东风，故国不堪回首月明中。雕栏玉砌应犹在，只是朱颜改。问君能有几多愁？恰似一江春水向东流。"
    },
    {
        "title": "游子吟",
        "author": "孟郊",
        "dynasty": "唐",
        "content": "慈母手中线，游子身上衣。临行密密缝，意恐迟迟归。谁言寸草心，报得三春晖。"
    }
]


@mcp.tool()
def search_poetry(keyword: str) -> str:
    """搜索包含指定关键词的古诗词

    Args:
        keyword: 搜索关键词，将在诗词标题和内容中查找

    Returns:
        str: 匹配的诗词信息（标题、作者、朝代、内容），未找到则返回提示信息
    """
    results = []

    for poem in POEMS:
        # 在标题和内容中搜索关键词（统一转小写做大小写无关匹配，
        # 兼容英文关键词如 "li bai" 的查找场景）
        if keyword.lower() in poem["title"].lower() or keyword.lower() in poem["content"].lower():
            results.append(
                f"《{poem['title']}》—— {poem['author']}（{poem['dynasty']}）\n"
                f"{poem['content']}"
            )

    if not results:
        return "未找到包含该关键词的诗词"

    return "\n\n".join(results)


@mcp.tool()
def random_poetry() -> str:
    """随机返回一首古诗词

    Returns:
        str: 随机诗词信息（标题、作者、朝代、内容）
    """
    poem = random.choice(POEMS)
    return f"《{poem['title']}》—— {poem['author']}（{poem['dynasty']}）\n{poem['content']}"


@mcp.tool()
def list_poets() -> str:
    """返回所有诗人列表（去重）

    Returns:
        str: 所有诗人及其朝代的列表
    """
    # 按诗人去重，记录朝代
    poet_dict = {}
    for poem in POEMS:
        author = poem["author"]
        dynasty = poem["dynasty"]
        if author not in poet_dict:
            poet_dict[author] = dynasty

    # 格式化输出
    poet_list = [f"{author}（{dynasty}）" for author, dynasty in sorted(poet_dict.items())]
    return "诗人列表：\n" + "\n".join(poet_list)


if __name__ == "__main__":
    # stdio 传输：从 stdin 读取请求、向 stdout 写响应，阻塞运行直到连接关闭；
    # 该模式配合客户端子进程拉起使用，适合本地集成（无端口、无网络暴露）
    mcp.run(transport="stdio")