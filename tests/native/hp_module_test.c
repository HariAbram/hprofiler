/*
 * hooks/common/hp_module.h, driven by tests/test_disasm_modules.py (plain,
 * and under ASan+UBSan when available): the module-id hash, the kernel
 * function table the CUDA and ROCm hooks name launches from, and the mod=
 * tag. Run with a file argument, it also prints that file's module id so
 * the test can compare it with the Python side (extractor.module_id).
 *
 *   names of any length come back whole (the hooks' tables used to cut
 *     them at 255 / 2047 bytes);
 *   unloading a module drops exactly its functions, so a reused handle
 *     never returns an earlier kernel's name or module;
 *   the returned name is a per-thread copy: a concurrent unload cannot free
 *     it under the caller (checked with threads);
 *   hp_elf64_image_size(): an in-memory ELF image's file size when the
 *     program headers or a segment lie after the section header table (CUDA
 *     cubins), .bss ignored, out-of-range headers rejected;
 *   hp_fatbin_size(): a CUDA fat binary is its header plus its payload (the
 *     hook used to save only the payload's length of bytes).
 * With a file argument it also prints the size it computes for that file.
 */
#include <pthread.h>
#include <stdio.h>
#include "../../hooks/common/hp_module.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d ", __FILE__, __LINE__); \
                                       fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static void *churn(void *arg) {
    (void)arg;
    char name[64];
    for (int i = 0; i < 20000; i++) {
        const void *mod = (const void *)(uintptr_t)(0x9000 + (i % 7));
        const void *fn = (const void *)(uintptr_t)(0x7000 + (i % 13));
        snprintf(name, sizeof(name), "churn_kernel_%d", i % 13);
        hp_module_add(mod, 0x100 + (uint64_t)(i % 7));
        hp_function_add(fn, mod, name);
        uint64_t m;
        const char *got = hp_function_lookup(fn, &m);
        if (got && strncmp(got, "churn_kernel_", 13) != 0) {
            fprintf(stderr, "FAIL corrupted name %s\n", got);
            __atomic_add_fetch(&fails, 1, __ATOMIC_RELAXED);
        }
        if (i % 5 == 0) hp_module_remove(mod);
    }
    return NULL;
}

static void put16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static void put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void put64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }

/* section header table at 1000..1192, program headers at 3000..3112, one
 * segment's file data up to 3500 (the file size), a .bss of 1 TiB */
static void make_elf(uint8_t *b) {
    memset(b, 0, 4096);
    memcpy(b, "\x7f" "ELF", 4);
    b[4] = 2; b[5] = 1;
    put64(b + 32, 3000); put16(b + 54, 56); put16(b + 56, 2);     /* phoff, phentsize, phnum */
    put64(b + 40, 1000); put16(b + 58, 64); put16(b + 60, 3);     /* shoff, shentsize, shnum */
    uint8_t *sh = b + 1000;
    put32(sh + 64 + 4, 1);  put64(sh + 64 + 24, 200);  put64(sh + 64 + 32, 500);         /* PROGBITS */
    put32(sh + 128 + 4, 8); put64(sh + 128 + 24, 2000); put64(sh + 128 + 32, 1ULL << 40); /* NOBITS */
    uint8_t *ph = b + 3000;
    put64(ph + 8, 64);        put64(ph + 32, 100);
    put64(ph + 56 + 8, 3200); put64(ph + 56 + 32, 300);
}

static void elf_size_checks(void) {
    static uint8_t b[4096];
    make_elf(b);
    CHECK(hp_elf64_image_size(b, 1 << 20) == 3500, "size %zu", hp_elf64_image_size(b, 1 << 20));
    CHECK(hp_elf64_image_size(b, 3400) == 0, "past the limit");
    make_elf(b);
    put64(b + 3000 + 56 + 32, ~0ULL);                             /* segment wraps around */
    CHECK(hp_elf64_image_size(b, 1 << 20) == 0, "wrapping segment");
    make_elf(b);
    put64(b + 40, ~0ULL - 10);                                    /* section headers out of range */
    CHECK(hp_elf64_image_size(b, 1 << 20) == 0, "bad shoff");
    make_elf(b);
    b[4] = 1;                                                     /* 32-bit: not handled */
    CHECK(hp_elf64_image_size(b, 1 << 20) == 0, "elf32");
    CHECK(hp_elf64_image_size("not an elf at all, just text.................................", 1 << 20) == 0,
          "not elf");

    uint8_t fb[32] = {0};
    put32(fb, 0xBA55ED50u); put16(fb + 4, 1); put16(fb + 6, 16); put64(fb + 8, 936);
    CHECK(hp_fatbin_size(fb, 1 << 20) == 952, "fatbin %zu", hp_fatbin_size(fb, 1 << 20));
    CHECK(hp_fatbin_size(fb, 900) == 0, "fatbin past the limit");
    put16(fb + 6, 8);
    CHECK(hp_fatbin_size(fb, 1 << 20) == 0, "fatbin header too short");
    put16(fb + 6, 16); put32(fb, 0x12345678u);
    CHECK(hp_fatbin_size(fb, 1 << 20) == 0, "not a fatbin");
}

int main(int argc, char **argv) {
    if (argc > 1) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) return 2;
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        unsigned char *buf = malloc(n > 0 ? (size_t)n : 1);
        if (fread(buf, 1, (size_t)n, f) != (size_t)n) return 2;
        fclose(f);
        printf("MOD %016llx\n", (unsigned long long)hp_fnv1a64(buf, (size_t)n));
        if (n >= 64 && memcmp(buf, "\x7f" "ELF", 4) == 0)
            printf("SIZE %zu\n", hp_elf64_image_size(buf, (size_t)n * 4));
        else if (n >= 16 && memcmp(buf, "\x50\xed\x55\xba", 4) == 0)
            printf("SIZE %zu\n", hp_fatbin_size(buf, (size_t)n * 4));
        free(buf);
        return 0;
    }

    /* FNV-1a 64 reference values */
    CHECK(hp_fnv1a64("", 0) == 0xcbf29ce484222325ULL, "empty");
    CHECK(hp_fnv1a64("a", 1) == 0xaf63dc4c8601ec8cULL, "a");
    CHECK(hp_fnv1a64("foobar", 6) == 0x85944171f73967e8ULL, "foobar");

    char tag[24];
    CHECK(strcmp(hp_mod_tag(tag, sizeof(tag), 0x0123456789abcdefULL), ",mod=0123456789abcdef") == 0, "tag %s", tag);
    CHECK(strcmp(hp_mod_tag(tag, sizeof(tag), 0), "") == 0, "zero tag");

    /* a 5000-character name from module A */
    static char longname[5001];
    memset(longname, 'k', 5000);
    longname[5000] = '\0';
    const void *A = (const void *)0x1000, *B = (const void *)0x2000;
    const void *f1 = (const void *)0x10, *f2 = (const void *)0x20;
    hp_module_add(A, 0xaaaa);
    hp_function_add(f1, A, longname);
    hp_function_add(f2, NULL, "no_module_kernel");
    uint64_t m = 1;
    const char *n1 = hp_function_lookup(f1, &m);
    CHECK(n1 && strlen(n1) == 5000 && m == 0xaaaa, "long name %zu mod %llx", n1 ? strlen(n1) : 0,
          (unsigned long long)m);
    const char *n2 = hp_function_lookup(f2, &m);
    CHECK(n2 && strcmp(n2, "no_module_kernel") == 0 && m == 0, "no-module kernel");
    CHECK(hp_function_lookup((const void *)0x30, &m) == NULL && m == 0, "unknown handle");

    /* the copy survives an unload; the unload forgets only A's functions */
    n1 = hp_function_lookup(f1, &m);
    hp_module_remove(A);
    CHECK(strlen(n1) == 5000, "per-thread copy changed after unload");
    CHECK(hp_function_lookup(f1, &m) == NULL, "f1 survived its module's unload");
    CHECK(hp_function_lookup(f2, &m) != NULL, "f2 dropped by another module's unload");

    /* the driver reuses f1's address for a kernel of module B */
    hp_module_add(B, 0xbbbb);
    hp_function_add(f1, B, "reused_kernel");
    n1 = hp_function_lookup(f1, &m);
    CHECK(n1 && strcmp(n1, "reused_kernel") == 0 && m == 0xbbbb, "reused handle: %s %llx", n1 ? n1 : "",
          (unsigned long long)m);
    /* a module loaded without captured code (id 0) gives its kernels no id */
    hp_module_add((const void *)0x3000, 0);
    hp_function_add((const void *)0x40, (const void *)0x3000, "uncaptured");
    hp_function_lookup((const void *)0x40, &m);
    CHECK(m == 0, "uncaptured module got an id");

    elf_size_checks();

    pthread_t t[4];
    for (int i = 0; i < 4; i++) pthread_create(&t[i], NULL, churn, NULL);
    for (int i = 0; i < 4; i++) pthread_join(t[i], NULL);

    printf("RESULT fails=%d\n", fails);
    return fails ? 1 : 0;
}
