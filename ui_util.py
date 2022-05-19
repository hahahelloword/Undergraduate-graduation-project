# 负责定义一些必要的函数接口，实现两个工程的结合
import inference

# 模型以及路径信息【【此处需要修改】】
model2pic = {"candy": "./styles/candy.jpg",
            "cubist": "./styles/cubist.jpg",
             "cuphead": "./styles/cuphead.jpg",
             "feathers": "./styles/feathers.jpg",
             "mona_lisa": "./mona_lisa.jpg",
             "picasso": "./styles/picasso.jpg",
             "wave": "./styles/wave.jpg",
             "starry_night":"./styles/starry_night",

             }
# 模型风格图片以及路径信息【【此处也需要修改】】
model2path = {"candy": "./checkpoints/select_model/candy_10000.pth",
              "cubist": "./checkpoints/select_model/cubist_2000.pth",
              "cuphead": "./checkpoints/select_model/cuphead_10000.pth",
              "feathers": "./checkpoints/select_model/feathers_2000.pth",
              "mona_lisa": "./checkpoints/select_model/mona_lisa_2000.pth",
              "picasso": "./checkpoints/select_model/picasso_2000.pth",
              "wave": "./checkpoints/select_model/wave_4000.pth",
              "starry_night": "./checkpoints/select_model/starry_night_10000.pth",

              }


input_path = "./styles/demo.jpg"  # 图片输入路径，这里是初始化占位
output_path = "images/outputs/res.jpg"  # 图片输出路径，这里也是占位，给UI界面取用
model_path = "./checkpoints/select_model/candy_10000.pth"  # 模型路径，第一个模型的有效路径

def load():
    global _model
    _model = inference.load_pth(model_path)

def run():
    # 调用主函数
    global output_path
    # 此处调用你的主函数
    print("Starting ...")
    print(input_path)
    print(model_path)
    output_path = inference.main('', input_path, model_path)
    print("DONE!")

if __name__ == '__main__':
    run()