# UI Layout Guardrails

## Tam yükseklikli splitter / resize hit-area kuralı

Tam panel yüksekliğinde bir `Button` veya `InvisibleButton`, yalnızca mouse hit-area'sı
olarak kullanılmalıdır. ImGui için yine de gerçek bir layout item olduğundan parent
cursor'ını ve aynı satırın yüksekliğini değiştirir.

Bu nedenle splitter çizilmeden önce içerik kolonunun başlangıç konumunu sakla:

```cpp
const float content_y = ImGui::GetCursorPosY();
const float content_x = sidebar_width + splitter_width;
```

Splitter çizildikten sonra content child'ı `SameLine()` zincirine bırakmadan açıkça
başlangıç konumuna yerleştir:

```cpp
ImGui::SetCursorPos(ImVec2(content_x, content_y));
ImGui::BeginChild("Content", ImVec2(0, 0), false);
```

`EndChild()` sonrasında cursor konumu child'ın alt kenarındadır; bu konum content
başlangıcı olarak kaydedilmemelidir. Aksi halde panel çerçevesi açılır, fakat child
içindeki tab, ikon ve butonlar ekran dışında veya sıfır/kliplenmiş alanda çizilir.

Bu kural `Properties`, sağ tool dock ve gelecekteki dockable editor panellerindeki
tüm splitter/resizer uygulamaları için geçerlidir.
