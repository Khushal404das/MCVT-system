<?php
declare(strict_types=1);

namespace Mcvt;

/**
 * Minimal YAML reader for config.yaml (block maps, block lists, inline [..] / {}
 * collections, quoted and plain scalars). PHP has no built-in YAML parser and
 * the project must not depend on Composer packages.
 */
final class Yaml
{
    /** @return array<string,mixed> */
    public static function parseFile(string $path): array
    {
        $text = @file_get_contents($path);
        if ($text === false) {
            throw new \RuntimeException("Cannot read $path");
        }
        return self::parse($text);
    }

    /** @return array<string,mixed> */
    public static function parse(string $text): array
    {
        $lines = [];
        foreach (preg_split('/\R/', $text) as $raw) {
            $line = self::stripComment($raw);
            if (trim($line) === '') {
                continue;
            }
            $indent = strlen($line) - strlen(ltrim($line, ' '));
            $lines[] = [$indent, trim($line)];
        }
        $i = 0;
        $result = self::parseBlock($lines, $i, 0);
        return is_array($result) ? $result : [];
    }

    private static function stripComment(string $line): string
    {
        $out = '';
        $quote = null;
        $len = strlen($line);
        for ($k = 0; $k < $len; $k++) {
            $c = $line[$k];
            if ($quote !== null) {
                if ($c === '\\' && $quote === '"' && $k + 1 < $len) {
                    $out .= $c . $line[++$k];
                    continue;
                }
                if ($c === $quote) {
                    $quote = null;
                }
            } elseif ($c === '"' || $c === "'") {
                $quote = $c;
            } elseif ($c === '#' && ($k === 0 || $line[$k - 1] === ' ')) {
                break;
            }
            $out .= $c;
        }
        return rtrim($out);
    }

    /** @param array<int,array{0:int,1:string}> $lines */
    private static function parseBlock(array $lines, int &$i, int $indent): mixed
    {
        if ($i >= count($lines)) {
            return null;
        }
        $isList = str_starts_with($lines[$i][1], '- ') || $lines[$i][1] === '-';
        $out = [];
        while ($i < count($lines)) {
            [$ind, $text] = $lines[$i];
            if ($ind < $indent) {
                break;
            }
            if ($ind > $indent) {
                $i++; // malformed / unexpected indentation - skip
                continue;
            }
            if ($isList) {
                if (!(str_starts_with($text, '- ') || $text === '-')) {
                    break;
                }
                $item = trim(substr($text, 1));
                $i++;
                if ($item === '') {
                    $out[] = self::parseBlock($lines, $i, $i < count($lines) ? $lines[$i][0] : $indent + 2);
                } elseif (self::isKeyValue($item)) {
                    // "- key: value" starts a map; following keys are indented deeper.
                    $childIndent = $indent + 2;
                    $virtual = [[$childIndent, $item]];
                    while ($i < count($lines) && $lines[$i][0] > $indent) {
                        $virtual[] = $lines[$i];
                        $i++;
                    }
                    $j = 0;
                    $out[] = self::parseBlock($virtual, $j, $childIndent);
                } else {
                    $out[] = self::scalar($item);
                }
                continue;
            }
            if (!self::isKeyValue($text)) {
                $i++;
                continue;
            }
            [$key, $value] = self::splitKey($text);
            $i++;
            if ($value === '') {
                if ($i < count($lines) && $lines[$i][0] > $indent) {
                    $out[$key] = self::parseBlock($lines, $i, $lines[$i][0]);
                } elseif ($i < count($lines) && $lines[$i][0] === $indent && str_starts_with($lines[$i][1], '- ')) {
                    $out[$key] = self::parseBlock($lines, $i, $indent);
                } else {
                    $out[$key] = null;
                }
            } else {
                $out[$key] = self::scalar($value);
            }
        }
        return $out;
    }

    private static function isKeyValue(string $text): bool
    {
        return (bool) preg_match('/^("[^"]*"|\'[^\']*\'|[^\s"\'][^:]*?)\s*:(\s|$)/', $text);
    }

    /** @return array{0:string,1:string} */
    private static function splitKey(string $text): array
    {
        preg_match('/^("[^"]*"|\'[^\']*\'|[^\s"\'][^:]*?)\s*:\s*(.*)$/', $text, $m);
        return [trim($m[1], "\"'"), trim($m[2])];
    }

    private static function scalar(string $v): mixed
    {
        $v = trim($v);
        if ($v === '' || $v === '~' || strtolower($v) === 'null') {
            return null;
        }
        if ($v === '{}') {
            return [];
        }
        if ($v[0] === '[' && str_ends_with($v, ']')) {
            $inner = trim(substr($v, 1, -1));
            if ($inner === '') {
                return [];
            }
            return array_map([self::class, 'scalar'], str_getcsv($inner, ',', '"', '\\'));
        }
        if ($v[0] === '"' && str_ends_with($v, '"')) {
            return stripcslashes(substr($v, 1, -1));
        }
        if ($v[0] === "'" && str_ends_with($v, "'")) {
            return str_replace("''", "'", substr($v, 1, -1));
        }
        $low = strtolower($v);
        if (in_array($low, ['true', 'yes', 'on'], true)) {
            return true;
        }
        if (in_array($low, ['false', 'no', 'off'], true)) {
            return false;
        }
        if (is_numeric($v)) {
            return $v + 0;
        }
        return $v;
    }
}
