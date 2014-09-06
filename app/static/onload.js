window.onload = function() {
    pagination_containers = $(".pagination_container");
    for (var i = 0; i < pagination_containers.length; i++) {
        var pagination = document.createElement("div");
        pagination.setAttribute("class", "pagination");
        pagination_containers[i].appendChild(pagination);
        var list = document.createElement("ul");
        pagination.appendChild(list);
        var page_no = parseInt(document.getElementById("page_no").getAttribute("value"));
        var start_page_no;
        if (page_no <= 5) {
            start_page_no = 1;
        } else {
            start_page_no = page_no - 4;
        }
        addPageLink(list, "前一页", "previousPage(); return false;");
        for(var j = 0; j < 10; j++) {
            var page_index = start_page_no + j;
            addPageLink(list, page_index, "setPage(this); return false;");
        }
        addPageLink(list, "后一页", "nextPage(); return false;");
    }
    document.getElementById("search_string").focus();
    document.getElementById("search_form").setAttribute("onSubmit", "return checkInput();");
}

function addPageLink(list, txt, onclick_script) {
    var list_item = document.createElement("li");
    list.appendChild(list_item);
    var link = document.createElement("a");
    list_item.appendChild(link);
    var txt_node = document.createTextNode(txt);
    link.appendChild(txt_node);
    link.setAttribute("onclick", onclick_script);
    link.setAttribute("href", "#");

    page_no_node = document.getElementById("page_no");
    page_no_str = page_no_node.getAttribute("value");
    page_no = parseInt(page_no_str) + 1;
    if (txt == page_no)
        list_item.setAttribute("class", "active");
}

function setPage(link) {
    text_node =link.childNodes[0];
    page_no = parseInt(text_node.nodeValue) - 1;
    forms = document.getElementsByTagName("form");
    document.getElementById("page_no").setAttribute("value", page_no);
    forms[0].submit();
}

function nextPage() {
    page_no_node = document.getElementById("page_no");
    page_no_str = page_no_node.getAttribute("value");
    page_no = parseInt(page_no_str) + 1;
    page_no_node.setAttribute("value", page_no);
    forms = document.getElementsByTagName("form");
    forms[0].submit();
}

function previousPage() {
    page_no_node = document.getElementById("page_no");
    page_no_str = page_no_node.getAttribute("value");
    page_no = parseInt(page_no_str) - 1;
    if (page_no < 0) page_no = 0;
    page_no_node.setAttribute("value", page_no);
    forms = document.getElementsByTagName("form");
    forms[0].submit();
}

function checkInput(){
    var search_str_node = document.getElementById('search_string');
    if(search_string.value == ""){
        alert("请输入检索字符串");
        return false;
    }
    return true;
}
