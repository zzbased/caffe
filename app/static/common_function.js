// 动画回到顶部
$(function(){
        //当滚动条的位置处于距顶部100像素以下时，跳转链接出现，否则消失
        $(function () {
            $(window).scroll(function(){
                if ($(window).scrollTop()>100){
                    $("#back-to-top").fadeIn(1500);
                }
                else
                {
                    $("#back-to-top").fadeOut(1500);
                }
            });
 
            //当点击跳转链接后，回到页面顶部位置
 
            $("#back-to-top").click(function(){
                $('body,html').animate({scrollTop:0},1000);
                return false;
            });
        });
    });

// 弹出提示窗显示地域码对照信息
function alert_area(){ 
    alert("0  无限定\n1  台湾省\n2  西藏\n3  青海省\n4  四川省\n5  广西\n6  江苏省\n7  吉林省\n8  山东省\n9  安徽省\n10 内蒙古\n11 山西省\n12 湖北省\n13 河北省\n14 江西省\n15 上海市\n16 浙江省\n17 澳门\n18 新疆\n19 福建省\n20 天津市\n21 北京市\n22 宁夏\n23 云南省\n24 辽宁省\n25 陕西省\n26 贵州省\n27 香港\n28 黑龙江\n29 海南省\n30 广东省\n31 重庆市\n32 河南省\n33 甘肃省\n34 湖南省\n63 未知", "地域码对照");
}
